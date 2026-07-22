# test_token_counter.py
# Description: Tests for token counting functionality
#
# Imports
import pytest

#
# Local Imports
from tldw_chatbook.Utils.token_counter import (
    count_tokens_chat_history,
    get_model_token_limit,
    estimate_remaining_tokens,
    format_token_display,
    TIKTOKEN_AVAILABLE,
)
#
########################################################################################################################
#
# Test Functions:


class TestTokenCounter:
    """Test token counting functionality"""

    def test_count_tokens_empty_history(self):
        """Test token counting with empty history"""
        result = count_tokens_chat_history([], model="gpt-3.5-turbo")
        assert result == 0

    def test_count_tokens_single_message(self):
        """Test token counting with single message"""
        history = [{"role": "user", "content": "Hello world"}]
        result = count_tokens_chat_history(history, model="gpt-3.5-turbo")
        assert result > 0  # Should count tokens for message + overhead

    def test_count_tokens_tuple_format(self):
        """Test token counting with tuple format history"""
        history = [("Hello from user", "Hello from assistant")]
        result = count_tokens_chat_history(history, model="gpt-3.5-turbo")
        assert result > 0

    def test_get_model_token_limit_known_model(self):
        """Test getting token limit for known models"""
        assert get_model_token_limit("gpt-4") == 8192
        assert get_model_token_limit("gpt-4-32k") == 32768
        assert get_model_token_limit("gpt-4-turbo") == 128000
        assert get_model_token_limit("claude-3-opus-20240229") == 200000

    def test_get_model_token_limit_unknown_model(self):
        """Test getting token limit for unknown model"""
        assert get_model_token_limit("unknown-model") == 4096  # Default

    def test_get_model_token_limit_by_prefix(self):
        """Test getting token limit by model prefix"""
        assert get_model_token_limit("gpt-4-some-variant") == 8192
        # Claude-3 models need exact match or will fall back to default
        assert get_model_token_limit("claude-3-opus-20240229") == 200000
        # Unknown claude variant falls back to default
        assert get_model_token_limit("claude-3-opus-custom") == 4096

    def test_estimate_remaining_tokens(self):
        """Test estimating remaining tokens"""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        used, limit, remaining = estimate_remaining_tokens(
            history, model="gpt-3.5-turbo", max_tokens_response=1000
        )
        assert used > 0
        assert limit == 16385  # gpt-3.5-turbo refreshed input window
        assert remaining < limit - 1000

    def test_get_model_token_limit_current_models(self):
        assert get_model_token_limit("gpt-4o", "openai") == 128000
        assert get_model_token_limit("claude-3-5-sonnet-20241022", "anthropic") == 200000
        assert get_model_token_limit("gemini-1.5-pro", "google") == 2097152
        assert get_model_token_limit("mistral-large", "mistral") == 128000

    def test_get_model_token_limit_prefers_capabilities_over_table(self):
        # A novel dated gpt-4o variant is not a table key; capabilities resolves it.
        assert get_model_token_limit("gpt-4o-2099-12-31", "openai") == 128000

    def test_get_model_token_limit_longest_prefix_wins(self):
        # "gpt-4" (8192) must not shadow a more specific match; a bare gpt-4 variant
        # with no capability pattern falls to the gpt-4 table prefix.
        assert get_model_token_limit("gpt-4-some-variant", "openai") == 8192

    def test_get_model_token_limit_anthropic_default_bumped(self):
        # Unknown modern Claude falls back to the 200k floor, not the stale 100k.
        assert get_model_token_limit("claude-99-future", "anthropic") == 200000

    def test_format_token_display_green(self):
        """Test token display formatting - green indicator"""
        result = format_token_display(100, 1000)
        assert "🟢" in result
        assert "100" in result
        assert "1,000" in result
        assert "10%" in result

    def test_format_token_display_yellow(self):
        """Test token display formatting - yellow warning"""
        result = format_token_display(850, 1000)
        assert "🟡" in result
        assert "85%" in result

    def test_format_token_display_red(self):
        """Test token display formatting - red danger"""
        result = format_token_display(980, 1000)
        assert "🔴" in result
        assert "98%" in result

    def test_format_token_display_zero_limit(self):
        """Test token display with zero limit (edge case)"""
        result = format_token_display(100, 0)
        assert "🟢" in result  # Should handle gracefully

    @pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_tiktoken_counting(self):
        """Test accurate token counting with tiktoken if available"""
        from tldw_chatbook.Utils.token_counter import count_tokens_tiktoken

        # Known text with predictable token count
        text = "Hello world"  # Should be 2 tokens for most models
        result = count_tokens_tiktoken(text, "gpt-3.5-turbo")
        assert result == 2

    def test_character_estimation_fallback(self):
        """Test character-based estimation when tiktoken not available"""
        # Test with a provider that uses character estimation
        history = [{"role": "user", "content": "A" * 100}]  # 100 characters
        result = count_tokens_chat_history(history, model="unknown", provider="unknown")

        # Should use default ratio of 0.25 tokens per char
        # Plus some overhead for message formatting
        assert result > 25  # At least 25 tokens for 100 chars
        assert result < 50  # But not too many

    def test_provider_specific_ratios(self):
        """Test provider-specific token/character ratios"""
        text = "A" * 100
        history = [{"role": "user", "content": text}]

        # Different providers might have different ratios
        openai_tokens = count_tokens_chat_history(history, provider="openai")
        google_tokens = count_tokens_chat_history(history, provider="google")

        # Both should return reasonable estimates
        assert openai_tokens > 0
        assert google_tokens > 0

    def test_system_prompt_in_estimation(self):
        """Test that system prompt is included in token estimation"""
        history = [{"role": "user", "content": "Hello"}]
        system_prompt = (
            "You are a helpful assistant with a very long system prompt " * 10
        )

        used_without, _, _ = estimate_remaining_tokens(history)
        used_with, _, _ = estimate_remaining_tokens(
            history, system_prompt=system_prompt
        )

        assert used_with > used_without  # System prompt should add tokens

    def test_mixed_history_formats(self):
        """Test handling of mixed history formats"""
        history = [
            ("User message", "Bot response"),  # Tuple format
            {"role": "user", "content": "Another message"},  # Dict format
            {"role": "assistant", "content": "Another response"},
        ]

        result = count_tokens_chat_history(history)
        assert result > 0  # Should handle mixed formats gracefully


#
# End of test_token_counter.py
########################################################################################################################
