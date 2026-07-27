"""Retained synchronous LLM worker adapter for non-Console destinations.

Native Console owns its worker, streaming, and cancellation lifecycle inside
``ChatScreen``. CCP and media analysis still use ``TldwCli.chat_wrapper`` for
non-streaming calls, so this adapter preserves that narrow contract without
publishing app-level worker or widget state.
"""

from typing import TYPE_CHECKING, Any

from ..Chat.Chat_Functions import chat as core_chat_function

if TYPE_CHECKING:
    from ..app import TldwCli


def chat_wrapper_function(
    app_instance: "TldwCli", strip_thinking_tags: bool = True, **kwargs: Any
) -> Any:
    """Run a retained non-Console LLM call without a root streaming bridge."""
    del app_instance
    if kwargs.get("streaming"):
        raise ValueError(
            "TldwCli.chat_wrapper no longer owns streaming calls; "
            "use the native Console provider gateway"
        )
    return core_chat_function(
        strip_thinking_tags=strip_thinking_tags,
        **kwargs,
    )
