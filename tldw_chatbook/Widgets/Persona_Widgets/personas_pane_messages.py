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


class EditCharacterRequested(Message):
    """User requested to edit the character.

    Relocated from the retired ``CCP_Widgets.ccp_character_card_widget``.
    """

    def __init__(self, character_id: str) -> None:
        self.character_id = character_id
        super().__init__()


class CharacterSaveRequested(Message):
    """User requested to save the character.

    Relocated from the retired ``CCP_Widgets.ccp_character_editor_widget``.
    """

    def __init__(self, character_data: Dict[str, Any]) -> None:
        super().__init__()
        self.character_data = character_data


class CharacterEditorCancelled(Message):
    """User cancelled character editing.

    Relocated from the retired ``CCP_Widgets.ccp_character_editor_widget``.
    """


class CharacterImageUploadRequested(Message):
    """User requested to choose an image for the active character editor."""


class CharacterImageRemoveRequested(Message):
    """User requested to remove the avatar image from the active character editor."""


class CharacterExpressionUploadRequested(Message):
    """User requested to choose an image for one expression-state slot
    (thinking/speaking/error) in the active character editor.

    Roleplay P3d-1 Task 4: distinct from ``CharacterImageUploadRequested``
    (the card's own avatar) - these write straight to the
    ``character_expression_images`` table, independent of the card's save.
    """

    def __init__(self, state: str) -> None:
        self.state = state
        super().__init__()


class CharacterExpressionClearRequested(Message):
    """User requested to clear one expression-state slot's image."""

    def __init__(self, state: str) -> None:
        self.state = state
        super().__init__()


class CharacterExpressionSetImportRequested(Message):
    """Roleplay P3d-2: import a whole expression set from a .zip."""


class CharacterExpressionSetExportRequested(Message):
    """Roleplay P3d-2: export the character's expression set to a .zip."""


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


class EditorContentChanged(Message):
    """An editor form received its first real user modification.

    Posted at most once per editing session (re-armed by each
    ``load_character``/``new_character``/``load_persona``/``new_persona``
    population) by the workbench editor widgets; the screen flips
    ``has_unsaved_changes`` on it.
    """


class PreviewReplyRequested(Message):
    """A test reply was requested from the preview-conversation pane."""

    def __init__(self, user_message: str) -> None:
        self.user_message = user_message
        super().__init__()


class PreviewResetRequested(Message):
    """The preview-conversation transcript was reset."""


class PreviewGreetingSelected(Message):
    """The user picked a greeting (index into the greetings list) to seed from."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index


class PreviewOpenInConsoleRequested(Message):
    """Open the preview-conversation transcript in Console."""


class PreviewConfigureProviderRequested(Message):
    """Open Settings > Providers & Models from the preview provider readout."""
