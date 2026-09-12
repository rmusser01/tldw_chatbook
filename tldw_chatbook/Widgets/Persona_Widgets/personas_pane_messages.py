"""Workbench pane messages added after the PR #506 foundation contract.

Kept separate from personas_messages.py so that file stays byte-identical to
the foundation PR until it merges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal
from uuid import UUID

from textual.message import Message


@dataclass(frozen=True, slots=True)
class VisualIdentityAssetMetadata:
    """Path-free metadata for one asset shown by the Personas browser."""

    asset_id: int
    expression_key: str
    original_label: str
    display_label: str
    content_type: str
    is_animated: bool


@dataclass(frozen=True, slots=True)
class VisualIdentityPackMetadata:
    """Path-free active-pack metadata safe to hand to a widget."""

    binding_id: int
    pack_id: int
    pack_version_id: int
    title: str
    source_kind: str
    default_expression_key: str
    assets: tuple[VisualIdentityAssetMetadata, ...]


class _VisualIdentityAssetRequested(Message):
    """Base for pack actions targeting one metadata-only asset."""

    def __init__(self, asset: VisualIdentityAssetMetadata) -> None:
        self.asset = asset
        super().__init__()


class VisualIdentityPackPreviewRequested(_VisualIdentityAssetRequested):
    """Ask PersonasScreen to decode the selected asset lazily."""


class VisualIdentityPackReplaceRequested(_VisualIdentityAssetRequested):
    """Ask PersonasScreen to stage a replacement for one asset."""


class VisualIdentityPackGenerateRequested(_VisualIdentityAssetRequested):
    """Ask PersonasScreen to stage a generated replacement for one asset."""


class VisualIdentityPackClearRequested(_VisualIdentityAssetRequested):
    """Ask PersonasScreen to stage removal of one asset."""


class VisualIdentityPackSaveRequested(Message):
    """Ask PersonasScreen to publish the widget's staged candidate."""

    def __init__(self, pack_id: int, pack_version_id: int) -> None:
        self.pack_id = pack_id
        self.pack_version_id = pack_version_id
        super().__init__()


class ConversationRowSelected(Message):
    """A saved conversation row in the inspector was selected."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        super().__init__()


class OlderConversationsRequested(Message):
    """Request the next or failed page of saved conversations."""


class ConversationsRequested(Message):
    """Request that the selected character's conversation browser be revealed."""


class ConversationSearchChanged(Message):
    """Request a fresh selected-character Keyword result generation."""

    def __init__(self, query: str) -> None:
        self.query = query
        super().__init__()


class EditCharacterRequested(Message):
    """User requested to edit the character.

    Relocated from the retired ``CCP_Widgets.ccp_character_card_widget``.
    """

    def __init__(self, character_id: str) -> None:
        self.character_id = character_id
        super().__init__()


CharacterTTSAction = Literal[
    "assign",
    "preview",
    "create",
    "edit",
    "remove",
    "dismiss_suggestion",
    "open_audio_cpp_settings",
    "open_speech_lab_apply",
    "generate_new_profile",
]


class CharacterTTSActionRequested(Message):
    """Request one profile action without carrying character authority."""

    def __init__(
        self,
        action: CharacterTTSAction,
        profile_id: UUID | None,
    ) -> None:
        if action not in {
            "assign",
            "preview",
            "create",
            "edit",
            "remove",
            "dismiss_suggestion",
            "open_audio_cpp_settings",
            "open_speech_lab_apply",
            "generate_new_profile",
        }:
            raise ValueError("invalid character TTS action")
        if profile_id is not None and type(profile_id) is not UUID:
            raise TypeError("profile_id must be a UUID")
        if (
            action
            in {
                "preview",
                "edit",
                "remove",
                "open_audio_cpp_settings",
                "open_speech_lab_apply",
                "generate_new_profile",
            }
            and profile_id is None
        ):
            raise ValueError("profile action requires profile_id")
        if action in {"create", "dismiss_suggestion"} and profile_id is not None:
            raise ValueError(f"{action} does not accept profile_id")
        self.action = action
        self.profile_id = profile_id
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


class CharacterAvatarGenerateRequested(Message):
    """User requested AI generation of a new avatar image for the active
    character editor.

    Image-gen P3: distinct from ``CharacterImageUploadRequested`` (a manual
    file pick) - this triggers a generation worker instead, staging the
    result into the editor the same way an uploaded avatar would.
    """


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


class CharacterExpressionGenerateRequested(Message):
    """User requested AI generation of one expression-state slot
    (thinking/speaking/error) in the active character editor.

    Image-gen P3: mirrors ``CharacterExpressionUploadRequested`` but triggers
    a generation worker instead of a file picker.
    """

    def __init__(self, state: str) -> None:
        self.state = state
        super().__init__()


class CharacterExpressionSetImportRequested(Message):
    """Roleplay P3d-2: import a whole expression set from a .zip."""


class CharacterExpressionSetExportRequested(Message):
    """Roleplay P3d-2: export the character's expression set to a .zip."""


class CharacterExpressionGenerateAllRequested(Message):
    """Image-gen P3: user requested AI generation of all expression-state
    slots (thinking/speaking/error) at once."""


class CharacterExpressionStylePickRequested(Message):
    """Image-gen P3: user requested to pick a style template used by
    subsequent avatar/expression AI generations in the active character
    editor.

    Mirrors the Console's own style picker (``ConsoleStylePickerModal``)
    but stores the resolved template on the screen instead of inserting a
    token into a composer draft - the character editor has no draft text
    for a token to live in.
    """


class EditPersonaProfileRequested(Message):
    """Edit was requested for the displayed persona."""

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
