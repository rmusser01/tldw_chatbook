"""Wire-format composition for local thinking controls (ADR-066)."""

from tldw_chatbook.Chat.console_provider_support import (
    build_local_thinking_payload_fields,
)


class TestLlamaCppFamily:
    def test_level_goes_into_chat_template_kwargs(self):
        fields = build_local_thinking_payload_fields(
            "llama_cpp", "low", None
        )
        assert fields == {"chat_template_kwargs": {"reasoning_effort": "low"}}

    def test_budget_goes_top_level(self):
        fields = build_local_thinking_payload_fields(
            "local_llamacpp", None, 2048
        )
        assert fields == {"reasoning_budget": 2048}

    def test_level_and_budget_together(self):
        fields = build_local_thinking_payload_fields(
            "local_llamafile", "xhigh", 4096
        )
        assert fields == {
            "chat_template_kwargs": {"reasoning_effort": "xhigh"},
            "reasoning_budget": 4096,
        }

    def test_none_effort_sends_verbatim_and_disables_thinking(self):
        fields = build_local_thinking_payload_fields(
            "local-llm", "none", None
        )
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
                "reasoning_budget": 1024,
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
        fields = build_local_thinking_payload_fields(
            "local_mlx_lm", "low", 2048
        )
        assert fields == {"chat_template_kwargs": {"reasoning_effort": "low"}}


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
