"""Workbench pane messages added after the PR #506 foundation contract.

Kept separate from personas_messages.py so that file stays byte-identical to
the foundation PR until it merges.
"""

from __future__ import annotations

from textual.message import Message


class ConversationRowSelected(Message):
    """A saved conversation row in the inspector was selected."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        super().__init__()
