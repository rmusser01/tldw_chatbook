"""Immutable state transitions for the Library Prompts adaptive reader.

This module is presentation-only. It references the existing Prompt editor and
block-editor states and leaves reads, writes, history, collections, and conflict
ownership with their existing services and controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from tldw_chatbook.Library.library_prompts_state import PromptEditorState
from tldw_chatbook.Widgets.Prompts.prompt_block_editor_state import (
    ValidationField,
    update_block,
)


PromptReaderMode = Literal["basic", "advanced", "info"]


@dataclass(frozen=True)
class PromptReaderRequest:
    """One Prompt detail request fenced against stale settlement."""

    destination: Literal["prompts"]
    prompt_id: int
    version: int
    generation: int


@dataclass(frozen=True)
class PromptReaderValidationTarget:
    """A validation failure and the work-pane control that owns recovery."""

    mode: Literal["basic", "advanced"]
    control_id: str
    message: str
    block_id: str | None = None
    block_field: ValidationField | None = None


@dataclass(frozen=True)
class PromptReaderState:
    """One lossless Prompts reader projection around an existing draft."""

    selected_id: int | None = None
    selected_version: int | None = None
    loaded_id: int | None = None
    loaded_version: int | None = None
    loaded_generation: int | None = None
    generation: int = 0
    mode: PromptReaderMode = "basic"
    draft: PromptEditorState | None = None
    dirty: bool = False
    loading: bool = False
    error: str | None = None
    unavailable: bool = False

    @property
    def loaded_actions_eligible(self) -> bool:
        """Whether identity-sensitive actions may target the loaded draft."""
        return (
            self.draft is not None
            and not self.loading
            and not self.unavailable
            and self.error is None
            and self.selected_id is not None
            and self.selected_id == self.loaded_id
            and self.selected_version == self.loaded_version
            and self.loaded_generation == self.generation
        )


def set_prompt_reader_mode(
    state: PromptReaderState,
    mode: PromptReaderMode,
) -> PromptReaderState:
    """Switch Basic, Advanced, or Info without replacing the draft."""
    if mode not in {"basic", "advanced", "info"}:
        raise ValueError("mode must be basic, advanced, or info.")
    return replace(state, mode=mode)


def select_prompt_for_reader(
    state: PromptReaderState,
    prompt_id: int,
    *,
    version: int,
) -> tuple[PromptReaderState, PromptReaderRequest]:
    """Select one Prompt and create its next fully fenced detail request."""
    if type(prompt_id) is not int or prompt_id < 1:
        raise ValueError("prompt_id must be a positive integer.")
    if type(version) is not int or version < 1:
        raise ValueError("version must be a positive integer.")
    generation = state.generation + 1
    request = PromptReaderRequest(
        destination="prompts",
        prompt_id=prompt_id,
        version=version,
        generation=generation,
    )
    return (
        replace(
            state,
            selected_id=prompt_id,
            selected_version=version,
            generation=generation,
            mode=state.mode if prompt_id == state.loaded_id else "basic",
            loading=True,
            error=None,
            unavailable=False,
        ),
        request,
    )


def _matches_prompt_reader_request(
    state: PromptReaderState,
    request: PromptReaderRequest,
) -> bool:
    return (
        request.destination == "prompts"
        and request.prompt_id == state.selected_id
        and request.version == state.selected_version
        and request.generation == state.generation
    )


def settle_prompt_reader_request(
    state: PromptReaderState,
    request: PromptReaderRequest,
    draft: PromptEditorState,
) -> PromptReaderState:
    """Adopt an existing editor draft only when every request fence matches."""
    if (
        not _matches_prompt_reader_request(state, request)
        or draft.prompt_id != request.prompt_id
        or draft.version != request.version
    ):
        return state
    return replace(
        state,
        loaded_id=request.prompt_id,
        loaded_version=request.version,
        loaded_generation=request.generation,
        draft=draft,
        dirty=False,
        loading=False,
        error=None,
        unavailable=False,
    )


def fail_prompt_reader_request(
    state: PromptReaderState,
    request: PromptReaderRequest,
    message: str,
) -> PromptReaderState:
    """Settle one matching detail failure while retaining the prior draft."""
    if not _matches_prompt_reader_request(state, request):
        return state
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be non-empty text.")
    return replace(
        state,
        loading=False,
        error=message.strip(),
        unavailable=state.draft is None,
    )


def update_prompt_reader_basic_lane(
    state: PromptReaderState,
    *,
    lane: Literal["system", "user"],
    content: str,
) -> PromptReaderState:
    """Edit one Basic lane while preserving every hidden Advanced value."""
    if lane not in {"system", "user"}:
        raise ValueError("lane must be system or user.")
    if not isinstance(content, str):
        raise TypeError("content must be text.")
    draft = state.draft
    if draft is None or draft.block_editor_state is None:
        raise ValueError("A structured Prompt draft is required.")
    lane_index = 0 if lane == "system" else 1
    blocks = draft.block_editor_state.definition.lanes[lane_index].blocks
    if len(blocks) != 1:
        raise ValueError("Basic editing requires exactly one block in each edited lane.")
    block_state = update_block(
        draft.block_editor_state,
        blocks[0].id,
        content=content,
    )
    if block_state is draft.block_editor_state:
        return state
    updated_draft = replace(
        draft,
        block_editor_state=block_state,
        artifact_type=block_state.artifact_type,
        compiled_system_preview=block_state.compiled_system,
        compiled_user_preview=block_state.compiled_user,
        system_prompt=block_state.compiled_system,
        user_prompt=block_state.compiled_user,
    )
    return replace(state, draft=updated_draft, dirty=True)


def validate_prompt_reader_draft(
    state: PromptReaderState,
) -> PromptReaderValidationTarget | None:
    """Return the owning mode and focus target for the first invalid field."""
    draft = state.draft
    if draft is None:
        return None
    if not draft.name.strip():
        return PromptReaderValidationTarget(
            mode="basic",
            control_id="library-prompt-name",
            message="Name is required; enter a Prompt name.",
        )
    block_state = draft.block_editor_state
    if block_state is None:
        return PromptReaderValidationTarget(
            mode="advanced",
            control_id="library-prompt-convert",
            message=draft.compatibility_reason or "This Prompt cannot be edited in Basic.",
        )
    if not block_state.issues:
        return None
    issue = block_state.issues[0]
    return PromptReaderValidationTarget(
        mode="advanced",
        control_id="library-prompt-block-editor",
        block_id=issue.block_id,
        block_field=issue.field,
        message=issue.message,
    )
