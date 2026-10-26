from .chat_conversation_scope_service import ChatConversationScopeService
from .chat_conversation_service import ChatConversationService
from .chat_loop_scope_service import ServerChatLoopScopeService
from .conversation_local_marks_service import (
    ConversationLocalMark,
    ConversationLocalMarksService,
)

__all__ = [
    "ChatConversationScopeService",
    "ChatConversationService",
    "ConversationLocalMark",
    "ConversationLocalMarksService",
    "ServerChatLoopScopeService",
    "ServerChatConversationService",
    "ServerChatLoopService",
]


def __getattr__(name):
    # Schema inspection must stay inert while recovery holds the startup fence.
    # Accessing either public service still imports its normal runtime boundary.
    from importlib import import_module

    modules = {
        "ServerChatConversationService": ".server_chat_conversation_service",
        "ServerChatLoopService": ".server_chat_loop_service",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
