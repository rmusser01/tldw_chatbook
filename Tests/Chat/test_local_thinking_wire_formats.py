"""Wire-format composition for local thinking controls (ADR-066)."""

from tldw_chatbook.Chat.console_provider_support import (
    build_local_thinking_payload_fields,
)


class TestLlamaCppFamily:
    def test_level_goes_into_chat_template_kwargs(self):
        fields = build_local_thinking_payload_fields("llama_cpp", "low", None)
        assert fields == {"chat_template_kwargs": {"reasoning_effort": "low"}}

    def test_budget_goes_top_level(self):
        fields = build_local_thinking_payload_fields("local_llamacpp", None, 2048)
        assert fields == {"reasoning_budget_tokens": 2048}

    def test_level_and_budget_together(self):
        fields = build_local_thinking_payload_fields("local_llamafile", "xhigh", 4096)
        assert fields == {
            "chat_template_kwargs": {"reasoning_effort": "xhigh"},
            "reasoning_budget_tokens": 4096,
        }

    def test_none_effort_sends_verbatim_and_disables_thinking(self):
        fields = build_local_thinking_payload_fields("local-llm", "none", None)
        assert fields == {
            "chat_template_kwargs": {
                "reasoning_effort": "none",
                "enable_thinking": False,
            }
        }

    def test_all_family_keys_share_the_shape(self):
        for key in ("llama_cpp", "local_llamacpp", "local_llamafile", "local-llm"):
            assert build_local_thinking_payload_fields(key, "low", 1024) == {
                "chat_template_kwargs": {"reasoning_effort": "low"},
                "reasoning_budget_tokens": 1024,
            }


class TestVllmFamily:
    def test_level_is_dual_placed(self):
        for key in ("vllm", "local_vllm"):
            assert build_local_thinking_payload_fields(key, "medium", None) == {
                "reasoning_effort": "medium",
                "chat_template_kwargs": {"reasoning_effort": "medium"},
            }

    def test_budget_is_dropped(self):
        assert build_local_thinking_payload_fields("vllm", None, 2048) == {}


class TestCustomOpenAI:
    def test_level_is_top_level_only(self):
        for key in ("custom-openai-api", "custom-openai-api-2"):
            assert build_local_thinking_payload_fields(key, "low", 2048) == {
                "reasoning_effort": "low"
            }


class TestMlx:
    def test_level_via_template_kwargs_budget_dropped(self):
        fields = build_local_thinking_payload_fields("local_mlx_lm", "low", 2048)
        assert fields == {"chat_template_kwargs": {"reasoning_effort": "low"}}


class TestTemplateSafeEfforts:
    """Live-verified (llama.cpp b10430 + Qwen3.8): strict chat templates
    validate reasoning_effort and raise on unknown values, so non-safe
    efforts must not be forwarded via chat_template_kwargs."""

    def test_minimal_effort_drops_template_kwargs_but_keeps_budget(self):
        fields = build_local_thinking_payload_fields("llama_cpp", "minimal", 2048)
        assert fields == {"reasoning_budget_tokens": 2048}

    def test_minimal_effort_drops_template_kwargs_on_mlx(self):
        fields = build_local_thinking_payload_fields("local_mlx_lm", "minimal", None)
        assert fields == {}

    def test_minimal_effort_on_vllm_keeps_top_level_verbatim(self):
        fields = build_local_thinking_payload_fields("vllm", "minimal", None)
        assert fields == {"reasoning_effort": "minimal"}
        assert "chat_template_kwargs" not in fields

    def test_high_effort_emitted_verbatim(self):
        # "high" is aliased to "xhigh" by the Qwen3.8 template and is safe.
        fields = build_local_thinking_payload_fields("llama_cpp", "high", None)
        assert fields == {"chat_template_kwargs": {"reasoning_effort": "high"}}

    def test_minimal_effort_on_custom_openai_stays_verbatim(self):
        fields = build_local_thinking_payload_fields(
            "custom-openai-api", "minimal", None
        )
        assert fields == {"reasoning_effort": "minimal"}


class TestUnknownKeysAndHygiene:
    def test_unknown_key_returns_empty(self):
        assert build_local_thinking_payload_fields("ollama", "low", 1024) == {}
        assert build_local_thinking_payload_fields("openai", "low", 1024) == {}

    def test_blank_effort_and_missing_budget_return_empty(self):
        assert build_local_thinking_payload_fields("llama_cpp", "", None) == {}
        assert build_local_thinking_payload_fields("llama_cpp", None, None) == {}

    def test_whitespace_effort_is_normalized(self):
        assert build_local_thinking_payload_fields("llama_cpp", "  low ", None) == {
            "chat_template_kwargs": {"reasoning_effort": "low"}
        }

    def test_non_int_budget_is_ignored(self):
        assert build_local_thinking_payload_fields("llama_cpp", None, "2048") == {}
