"""Chat services are explicit imports; pure storage models initialize no runtime."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chat_conversation_scope_service import (
        ChatConversationScopeService as ChatConversationScopeService,
    )
    from .chat_conversation_service import (
        ChatConversationService as ChatConversationService,
    )
    from .chat_loop_scope_service import (
        ServerChatLoopScopeService as ServerChatLoopScopeService,
    )
    from .conversation_local_marks_service import (
        ConversationLocalMark as ConversationLocalMark,
    )
    from .conversation_local_marks_service import (
        ConversationLocalMarksService as ConversationLocalMarksService,
    )
    from .server_chat_conversation_service import (
        ServerChatConversationService as ServerChatConversationService,
    )
    from .server_chat_loop_service import ServerChatLoopService as ServerChatLoopService

_EXPORTS = {
    "ChatConversationScopeService": "chat_conversation_scope_service",
    "ChatConversationService": "chat_conversation_service",
    "ConversationLocalMark": "conversation_local_marks_service",
    "ConversationLocalMarksService": "conversation_local_marks_service",
    "ServerChatLoopScopeService": "chat_loop_scope_service",
    "ServerChatConversationService": "server_chat_conversation_service",
    "ServerChatLoopService": "server_chat_loop_service",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load the existing public service only when explicitly requested."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("." + module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
