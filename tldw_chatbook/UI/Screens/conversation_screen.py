"""Conversation/Character screen implementation.

This module re-exports the CCPScreen for backwards compatibility.
The actual implementation is in ccp_screen.py following Textual best practices.
"""

from .ccp_screen import CCPScreen as ConversationScreen

__all__ = ['ConversationScreen']