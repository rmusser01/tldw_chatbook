"""Dependency-light public exports for Prompt management services."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .local_prompt_service import LocalPromptService as LocalPromptService
    from .prompt_chatbook_scope_service import (
        PromptChatbookBackend as PromptChatbookBackend,
        PromptChatbookScopeService as PromptChatbookScopeService,
    )
    from .server_prompt_service import ServerPromptService as ServerPromptService

__all__ = [
    "LocalPromptService",
    "PromptChatbookBackend",
    "PromptChatbookScopeService",
    "ServerPromptService",
]

_LAZY_EXPORTS = {
    "LocalPromptService": ("local_prompt_service", "LocalPromptService"),
    "PromptChatbookBackend": (
        "prompt_chatbook_scope_service",
        "PromptChatbookBackend",
    ),
    "PromptChatbookScopeService": (
        "prompt_chatbook_scope_service",
        "PromptChatbookScopeService",
    ),
    "ServerPromptService": ("server_prompt_service", "ServerPromptService"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = value
    return value
