"""Workbench pane messages added after the PR #506 foundation contract.

Kept separate from personas_messages.py so that file stays byte-identical to
the foundation PR until it merges.
"""

from __future__ import annotations

from typing import Any, Dict

from textual.message import Message


class ConversationRowSelected(Message):
    """A saved conversation row in the inspector was selected."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        super().__init__()


class EditPersonaRequested(Message):
    """Edit was requested for the displayed persona profile."""

    def __init__(self, persona_id: str) -> None:
        self.persona_id = persona_id
        super().__init__()


class PersonaProfileSaveRequested(Message):
    """The persona editor form was submitted."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data
        super().__init__()


class PersonaProfileEditCancelled(Message):
    """The persona editor form was cancelled."""


class PreviewReplyRequested(Message):
    """A test reply was requested from the preview-conversation pane."""

    def __init__(self, user_message: str) -> None:
        self.user_message = user_message
        super().__init__()


class PreviewResetRequested(Message):
    """The preview-conversation transcript was reset."""


class PreviewOpenInConsoleRequested(Message):
    """Open the preview-conversation transcript in Console."""
