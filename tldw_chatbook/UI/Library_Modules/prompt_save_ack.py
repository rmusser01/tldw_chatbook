"""Submitted-only acknowledgement for the Library Prompt editor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from textual.css.query import QueryError

from ...Library.library_prompts_state import build_prompt_editor_state
from ...Widgets.Library.library_prompt_work_pane import LibraryPromptWorkPane
from ...Widgets.Prompts.prompt_block_editor_state import PromptBlockEditorState
from .screen_constants import LIBRARY_PROMPT_SAVE_STATUS_COPY

if TYPE_CHECKING:
    from .library_prompts_controller import LibraryPromptsController


async def acknowledge_prompt_save(
    controller: LibraryPromptsController,
    result: object,
    *,
    prompt_id: int | None,
    is_create: bool,
    submitted_fields: tuple[str, str, str, str, str, str],
    submitted_state: PromptBlockEditorState | None,
    prepared_state: PromptBlockEditorState | None,
    save_fields: Mapping[str, Any],
) -> None:
    """Adopt the committed record while retaining newer accepted controls.

    Args:
        controller: Existing destination draft and service owner.
        result: Successful result from the existing Prompt scope service.
        prompt_id: Submitted update identity, absent for creation.
        is_create: Whether this Save created the active Prompt.
        submitted_fields: Raw metadata and compiled text at submission.
        submitted_state: Immutable submitted block working copy.
        prepared_state: Validated artifact state submitted to persistence.
        save_fields: Existing normalized service request.
    """
    owner = controller
    # Removing a prior conflict affordance can await a DOM mutation. Capture
    # current input only after it, rather than treating that await as a fence.
    await owner._apply_library_prompt_save_outcome("ok")
    try:
        canvas = owner.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)
    except QueryError:
        canvas = None
    current_state = owner._library_prompt_block_state
    if canvas is not None and current_state is not None:
        current_state = canvas.capture_live_block_state()
        owner._library_prompt_block_state = current_state
    current_fields = owner._read_library_prompt_editor_fields()
    newer_blocks = (
        current_state is not None
        and submitted_state is not None
        and current_state.definition != submitted_state.definition
    )
    result_data = result if isinstance(result, Mapping) else {}
    new_id = result_data.get("local_id", 1) if is_create else prompt_id
    version = result_data.get("version")
    if version is None:
        version = 1 if is_create else (owner._library_prompt_version or 0) + 1
    detail = dict(owner._library_prompt_detail or {})
    detail.update(result_data)
    detail.update(
        {
            key: save_fields[key]
            for key in ("name", "author", "details", "system_prompt", "user_prompt")
        }
    )
    detail.update(id=new_id, version=version)
    if prepared_state is not None:
        detail.update(
            artifact_type=prepared_state.artifact_type,
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=save_fields["prompt_definition"],
        )
    if "keywords" not in result_data and save_fields.get("keywords") is not None:
        detail["keywords"] = save_fields["keywords"]
    history_was_open = bool(
        owner._library_prompt_history_state is not None
        and owner._library_prompt_history_state.is_open
    )
    if is_create:
        armed = owner._library_prompt_editor_armed
        owner._adopt_library_prompt_persisted_detail(
            detail,
            status=LIBRARY_PROMPT_SAVE_STATUS_COPY["ok"],
            open_history=history_was_open,
        )
        owner._library_prompt_editor_armed = armed
        owner._load_library_prompt_memberships()
    else:
        owner._library_prompt_detail = detail
        owner._library_prompt_original_name = save_fields["name"]
        owner._library_prompt_version = version
        owner._library_prompt_detached_structured = False
    owner._selected_prompt_id = new_id
    owner._library_prompt_loaded_id = new_id
    owner._library_prompt_detail_selected_name = save_fields["name"]
    owner._library_prompt_detail_loading = False
    owner._library_prompt_detail_error = ""
    owner._library_prompt_detail_retryable = False
    saved = build_prompt_editor_state(
        detail, capabilities=owner._library_prompt_capabilities
    )
    state = current_state if newer_blocks else saved.block_editor_state
    if newer_blocks and state is not None and saved.block_editor_state is not None:
        saved_blocks = {
            block.id: block
            for lane in saved.block_editor_state.definition.lanes
            for block in lane.blocks
        }
        state = replace(
            state,
            dirty_block_ids=frozenset(
                block.id
                for lane in state.definition.lanes
                for block in lane.blocks
                if block != saved_blocks.get(block.id)
            ),
        )
    owner._library_prompt_block_state = state
    if canvas is not None:
        working = replace(saved, block_editor_state=state)
        state = await canvas.sync_saved_editor_state(working)
        owner._library_prompt_block_state = state
        current_fields = owner._read_library_prompt_editor_fields()
    owner._library_prompt_dirty = current_fields != submitted_fields or bool(
        state is not None
        and submitted_state is not None
        and state.definition != submitted_state.definition
    )
    owner._sync_library_prompt_save_action_widgets()
    owner._set_library_prompt_discard_enabled(owner._library_prompt_dirty)
    owner._update_library_prompt_meta_static()
    if owner._library_prompt_dirty:
        owner._update_library_prompt_status_static(
            "Submitted Prompt saved; newer edits are still unsaved."
        )
    if not is_create:
        owner._initialize_library_prompt_history(detail, open_history=history_was_open)
    else:
        owner._request_library_prompts_browse(
            owner._library_prompt_browse_controller.mutation_refresh_scope,
            focus_identity=owner._library_prompts_focus_identity(),
        )
        owner._refresh_local_source_snapshot()
