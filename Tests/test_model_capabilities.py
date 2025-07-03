# Tests/test_model_capabilities.py
# Description: Unit tests for the ModelCapabilities system
#
# Imports
#
# Standard Library
import pytest
from unittest.mock import patch, MagicMock
import re

# Local Imports
from tldw_chatbook.model_capabilities import (
    ModelCapabilities, 
    get_model_capabilities, 
    is_vision_capable,
    reload_capabilities
)

# Test marker
pytestmark = pytest.mark.unit

#
#######################################################################################################################
#
# Test Fixtures
#

@pytest.fixture
def empty_config():
    """Empty configuration for testing defaults."""
    return {}

@pytest.fixture
def custom_config():
    """Custom configuration for testing."""
    return {
        "models": {
            "test-model-1": {"vision": True, "max_images": 5},
            "test-model-2": {"vision": False},
            "custom-vision-model": {"vision": True, "custom_field": "test"}
        },
        "patterns": {
            "TestProvider": [
                {"pattern": r"^test-vision-.*", "vision": True},
                {"pattern": r"^test-text-.*", "vision": False}
            ],
            "CustomProvider": [
                {"pattern": r".*-vision$", "vision": True, "max_images": 3}
            ]
        },
        "defaults": {
            "unknown_models_vision": True,
            "log_unknown_models": False
        }
    }

@pytest.fixture
def model_capabilities_empty():
    """ModelCapabilities instance with empty config."""
    return ModelCapabilities({})

@pytest.fixture
def model_capabilities_custom(custom_config):
    """ModelCapabilities instance with custom config."""
    return ModelCapabilities(custom_config)

#
#######################################################################################################################
#
# Basic Functionality Tests
#

class TestModelCapabilitiesBasics:
    """Test basic functionality of ModelCapabilities."""
    
    def test_initialization_with_empty_config(self, model_capabilities_empty):
        """Test that ModelCapabilities initializes with default values when config is empty."""
        # Should have default mappings
        assert len(model_capabilities_empty.direct_mappings) > 0
        assert "gpt-4-vision-preview" in model_capabilities_empty.direct_mappings
        
        # Should have default patterns
        assert len(model_capabilities_empty.pattern_configs) > 0
        assert "OpenAI" in model_capabilities_empty.pattern_configs
    
    def test_initialization_with_custom_config(self, model_capabilities_custom):
        """Test that ModelCapabilities properly loads custom configuration."""
        # Should have custom mappings
        assert "test-model-1" in model_capabilities_custom.direct_mappings
        assert model_capabilities_custom.direct_mappings["test-model-1"]["vision"] is True
        
        # Should have custom patterns
        assert "TestProvider" in model_capabilities_custom.pattern_configs
        
        # Should have custom defaults
        assert model_capabilities_custom.defaults["unknown_models_vision"] is True
    
    def test_direct_mapping_lookup(self, model_capabilities_custom):
        """Test that direct model mappings work correctly."""
        # Known vision model
        assert model_capabilities_custom.is_vision_capable("AnyProvider", "test-model-1") is True
        
        # Known non-vision model
        assert model_capabilities_custom.is_vision_capable("AnyProvider", "test-model-2") is False
        
        # Custom field in capabilities
        caps = model_capabilities_custom.get_model_capabilities("AnyProvider", "custom-vision-model")
        assert caps["vision"] is True
        assert caps["custom_field"] == "test"

#
#######################################################################################################################
#
# Pattern Matching Tests
#

class TestPatternMatching:
    """Test pattern-based model detection."""
    
    def test_pattern_matching_basic(self, model_capabilities_custom):
        """Test basic pattern matching."""
        # Should match pattern
        assert model_capabilities_custom.is_vision_capable("TestProvider", "test-vision-123") is True
        assert model_capabilities_custom.is_vision_capable("TestProvider", "test-text-abc") is False
        
        # Pattern with suffix (.*-vision$ matches anything ending with -vision)
        assert model_capabilities_custom.is_vision_capable("CustomProvider", "model-vision") is True
        assert model_capabilities_custom.is_vision_capable("CustomProvider", "test-vision") is True
        
        # This should NOT match because it doesn't end with -vision
        # But there might be a fallback to unknown_models_vision default
        # Let's check what the actual result is
        result = model_capabilities_custom.is_vision_capable("CustomProvider", "vision-model")
        # Since unknown_models_vision is True in custom_config, it falls back to True
        assert result is True  # It's True because of the default, not the pattern
    
    def test_pattern_matching_case_insensitive(self, model_capabilities_custom):
        """Test that pattern matching is case-insensitive."""
        assert model_capabilities_custom.is_vision_capable("TestProvider", "TEST-VISION-ABC") is True
        assert model_capabilities_custom.is_vision_capable("TestProvider", "Test-Vision-XYZ") is True
    
    def test_pattern_priority(self, model_capabilities_custom):
        """Test that direct mappings take priority over patterns."""
        # Add a pattern that would match but direct mapping should win
        model_capabilities_custom.add_model_capability("test-vision-direct", {"vision": False})
        
        # Even though it matches the vision pattern, direct mapping says no vision
        assert model_capabilities_custom.is_vision_capable("TestProvider", "test-vision-direct") is False
    
    def test_invalid_pattern_handling(self):
        """Test handling of invalid regex patterns."""
        config = {
            "patterns": {
                "BadProvider": [
                    {"pattern": r"[invalid(regex", "vision": True}  # Invalid regex
                ]
            }
        }
        # Should not crash, just skip the bad pattern
        caps = ModelCapabilities(config)
        assert caps.is_vision_capable("BadProvider", "any-model") is False

#
#######################################################################################################################
#
# Default Model Tests
#

class TestDefaultModels:
    """Test detection of known default models."""
    
    def test_openai_models(self, model_capabilities_empty):
        """Test OpenAI model detection."""
        # Direct mappings
        assert model_capabilities_empty.is_vision_capable("OpenAI", "gpt-4-vision-preview") is True
        assert model_capabilities_empty.is_vision_capable("OpenAI", "gpt-4-turbo") is True
        assert model_capabilities_empty.is_vision_capable("OpenAI", "gpt-4o") is True
        assert model_capabilities_empty.is_vision_capable("OpenAI", "gpt-4o-mini") is True
        
        # Pattern matching
        assert model_capabilities_empty.is_vision_capable("OpenAI", "gpt-4-vision-2024") is True
        assert model_capabilities_empty.is_vision_capable("OpenAI", "gpt-40-custom") is True
        
        # Non-vision models
        assert model_capabilities_empty.is_vision_capable("OpenAI", "gpt-3.5-turbo") is False
    
    def test_anthropic_models(self, model_capabilities_empty):
        """Test Anthropic model detection."""
        # Direct mappings
        assert model_capabilities_empty.is_vision_capable("Anthropic", "claude-3-opus-20240229") is True
        assert model_capabilities_empty.is_vision_capable("Anthropic", "claude-3-sonnet-20240229") is True
        
        # Pattern matching
        assert model_capabilities_empty.is_vision_capable("Anthropic", "claude-3-new-model") is True
        assert model_capabilities_empty.is_vision_capable("Anthropic", "claude-opus-4-20250514") is True
        
        # Non-vision models
        assert model_capabilities_empty.is_vision_capable("Anthropic", "claude-2.1") is False
    
    def test_google_models(self, model_capabilities_empty):
        """Test Google model detection."""
        # Direct mappings
        assert model_capabilities_empty.is_vision_capable("Google", "gemini-pro-vision") is True
        assert model_capabilities_empty.is_vision_capable("Google", "gemini-1.5-pro") is True
        
        # Pattern matching
        assert model_capabilities_empty.is_vision_capable("Google", "gemini-2.0-flash") is True
        assert model_capabilities_empty.is_vision_capable("Google", "gemini-ultra-vision") is True

#
#######################################################################################################################
#
# Unknown Model Handling Tests
#

class TestUnknownModels:
    """Test handling of unknown models."""
    
    def test_unknown_model_default_false(self, model_capabilities_empty):
        """Test that unknown models default to no vision."""
        assert model_capabilities_empty.is_vision_capable("UnknownProvider", "unknown-model") is False
    
    def test_unknown_model_custom_default(self, model_capabilities_custom):
        """Test custom default for unknown models."""
        # This config has unknown_models_vision = True
        assert model_capabilities_custom.is_vision_capable("UnknownProvider", "unknown-model") is True
    
    def test_unknown_provider_known_model(self, model_capabilities_custom):
        """Test that direct mappings work regardless of provider."""
        # test-model-1 is in direct mappings
        assert model_capabilities_custom.is_vision_capable("CompletelyUnknownProvider", "test-model-1") is True

#
#######################################################################################################################
#
# Caching and Performance Tests
#

class TestCachingAndPerformance:
    """Test caching functionality."""
    
    def test_capability_caching(self, model_capabilities_custom):
        """Test that capabilities are cached properly."""
        # First call
        caps1 = model_capabilities_custom.get_model_capabilities("TestProvider", "test-model")
        
        # Second call should return cached result
        caps2 = model_capabilities_custom.get_model_capabilities("TestProvider", "test-model")
        
        # Should be the same dict
        assert caps1 is caps2
    
    def test_is_vision_capable_caching(self, model_capabilities_custom):
        """Test that is_vision_capable uses LRU cache."""
        # The method has @lru_cache decorator
        cache_info = model_capabilities_custom.is_vision_capable.cache_info()
        initial_hits = cache_info.hits
        
        # Call multiple times
        for _ in range(5):
            model_capabilities_custom.is_vision_capable("TestProvider", "cached-model")
        
        # Should have cache hits
        cache_info = model_capabilities_custom.is_vision_capable.cache_info()
        assert cache_info.hits > initial_hits
    
    def test_cache_clearing(self, model_capabilities_custom):
        """Test cache clearing functionality."""
        # Add to cache
        model_capabilities_custom.is_vision_capable("TestProvider", "cached-model")
        model_capabilities_custom.get_model_capabilities("TestProvider", "cached-model")
        
        # Clear cache
        model_capabilities_custom.clear_cache()
        
        # Cache should be empty
        assert len(model_capabilities_custom._capability_cache) == 0
        assert model_capabilities_custom.is_vision_capable.cache_info().currsize == 0

#
#######################################################################################################################
#
# Module-level Function Tests
#

class TestModuleFunctions:
    """Test module-level convenience functions."""
    
    @patch('tldw_chatbook.model_capabilities.get_cli_setting')
    def test_get_model_capabilities(self, mock_get_setting):
        """Test get_model_capabilities function."""
        mock_get_setting.return_value = {}
        
        # Should return singleton instance
        caps1 = get_model_capabilities()
        caps2 = get_model_capabilities()
        assert caps1 is caps2
    
    @patch('tldw_chatbook.model_capabilities.get_cli_setting')
    def test_is_vision_capable_function(self, mock_get_setting):
        """Test module-level is_vision_capable function."""
        mock_get_setting.return_value = {
            "models": {"test-model": {"vision": True}}
        }
        
        # Clear any existing instance
        import tldw_chatbook.model_capabilities
        tldw_chatbook.model_capabilities._global_capabilities = None
        
        # Should work
        assert is_vision_capable("AnyProvider", "test-model") is True
    
    @patch('tldw_chatbook.model_capabilities.get_cli_setting')
    def test_reload_capabilities(self, mock_get_setting):
        """Test reloading capabilities."""
        mock_get_setting.return_value = {}
        
        # Get initial instance
        caps1 = get_model_capabilities()
        
        # Reload
        reload_capabilities()
        
        # Should be new instance
        caps2 = get_model_capabilities()
        assert caps1 is not caps2

#
#######################################################################################################################
#
# Edge Cases and Error Handling
#

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_pattern_list(self):
        """Test provider with empty pattern list."""
        config = {
            "patterns": {
                "EmptyProvider": []
            }
        }
        caps = ModelCapabilities(config)
        assert caps.is_vision_capable("EmptyProvider", "any-model") is False
    
    def test_malformed_config(self):
        """Test handling of malformed configuration."""
        config = {
            "models": "not-a-dict",  # Wrong type
            "patterns": {
                "Provider": "not-a-list"  # Wrong type
            }
        }
        # Should not crash
        caps = ModelCapabilities(config)
        assert caps.is_vision_capable("Provider", "model") is False
    
    def test_add_model_capability(self, model_capabilities_custom):
        """Test dynamically adding model capabilities."""
        # Add new model
        model_capabilities_custom.add_model_capability("dynamic-model", {"vision": True, "custom": "value"})
        
        # Should be detected
        assert model_capabilities_custom.is_vision_capable("AnyProvider", "dynamic-model") is True
        
        # Should have custom field
        caps = model_capabilities_custom.get_model_capabilities("AnyProvider", "dynamic-model")
        assert caps["custom"] == "value"
    
    def test_list_vision_models(self, model_capabilities_custom):
        """Test listing known vision models."""
        vision_models = model_capabilities_custom.list_vision_models()
        
        # Should include direct mappings with vision=True
        assert "test-model-1" in vision_models
        assert "custom-vision-model" in vision_models
        
        # Should not include non-vision models
        assert "test-model-2" not in vision_models
        
        # Should be sorted
        assert vision_models == sorted(vision_models)

#
# End of test_model_capabilities.py
#######################################################################################################################