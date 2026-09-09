"""Fixture-only bridge to the existing DeepSeek adapter and usage contract."""

from collections.abc import Mapping
from typing import Any


def complete_deepseek(**kwargs: Any) -> dict:
    """Run one tool-free, non-thinking request inside the auxiliary gateway scope.

    The native adapter owns HTTP, credentials, errors, and sensitive logging.
    DeepSeek cache counters are converted to the shared usage input format.
    """
    from tldw_chatbook.LLM_Calls.LLM_API_Calls import chat_with_deepseek

    if kwargs.get("api_endpoint") != "deepseek":
        raise ValueError("invalid_provider_identity")
    response = chat_with_deepseek(
        input_data=kwargs["messages_payload"],
        model=kwargs["model"],
        api_key=kwargs["api_key"],
        api_base_url=kwargs["api_base_url"],
        system_message=kwargs.get("system_message"),
        temp=kwargs.get("temp"),
        topp=kwargs.get("topp"),
        max_tokens=kwargs["max_tokens"],
        streaming=False,
        thinking_mode="disabled",
        response_format=kwargs.get("response_format"),
    )
    if not isinstance(response, Mapping):
        raise TypeError("invalid_provider_response")
    result = dict(response)
    usage = response.get("usage")
    required = (
        "prompt_tokens",
        "completion_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    if (
        not isinstance(usage, Mapping)
        or any(
            type(usage.get(field)) is not int or usage[field] < 0 for field in required
        )
        or (
            usage["prompt_cache_hit_tokens"] + usage["prompt_cache_miss_tokens"]
            != usage["prompt_tokens"]
        )
    ):
        result["usage"] = None
    else:
        result["usage"] = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "prompt_tokens_details": {
                "cached_tokens": usage["prompt_cache_hit_tokens"]
            },
        }
    return result
