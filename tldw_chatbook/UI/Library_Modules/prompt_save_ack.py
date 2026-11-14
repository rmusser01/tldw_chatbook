"""Prompt save submission and submitted-only acknowledgement for Library."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from textual.css.query import QueryError

from ...Library.library_prompts_state import build_prompt_editor_state
from ...Widgets.Library.library_prompt_work_pane import LibraryPromptWorkPane
from ...Widgets.Prompts.prompt_block_editor_state import PromptBlockEditorState
from ...Widgets.Prompts.prompt_mixin_editor import PromptMixinEditor
from .screen_constants import LIBRARY_PROMPT_SAVE_STATUS_COPY

if TYPE_CHECKING:
    from ...Prompt_Management.prompt_artifact_models import ArtifactType
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
    submitted_mixin_draft: tuple[str, str, str] | None = None,
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
        submitted_mixin_draft: Raw literal fields captured before Save.
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
    current_mixin = None
    if submitted_mixin_draft is not None:
        editors = owner.query(PromptMixinEditor)
        if not editors:
            return
        current_mixin = editors.first().read_draft()
    newer_mixin = current_mixin != submitted_mixin_draft
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
    if submitted_mixin_draft is not None:
        detail.update(
            {
                key: save_fields[key]
                for key in (
                    "artifact_type",
                    "prompt_format",
                    "prompt_schema_version",
                    "prompt_definition",
                )
            }
        )
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
    if submitted_mixin_draft is not None:
        owner._library_prompt_mixin_draft = current_mixin if newer_mixin else None
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
        working = replace(
            saved,
            block_editor_state=state,
            mixin_draft=owner._library_prompt_mixin_draft,
        )
        state = await canvas.sync_saved_editor_state(working)
        owner._library_prompt_block_state = state
        current_fields = owner._read_library_prompt_editor_fields()
    owner._library_prompt_dirty = (
        newer_mixin
        or current_fields != submitted_fields
        or bool(
            state is not None
            and submitted_state is not None
            and state.definition != submitted_state.definition
        )
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


async def save_prompt(
    self: LibraryPromptsController,
    *,
    target_artifact_type: ArtifactType | None = None,
    save_as_new: bool = False,
    source_globals: Mapping[str, Any],
) -> None:
    """Submit and acknowledge a save through the original controller seams."""
    if self._library_prompts_mutation_in_flight:
        return
    if self._library_prompts_view != "editor":
        return
    selected_prompt_id = self._selected_prompt_id
    prompt_id = None if save_as_new else selected_prompt_id
    is_create = prompt_id is None
    if self._library_prompt_block_state is not None:
        try:
            canvas = self.query_one(
                "#library-prompt-work-pane", source_globals["LibraryPromptWorkPane"]
            )
        except source_globals["QueryError"]:
            pass
        else:
            self._library_prompt_block_state = canvas.capture_live_block_state()
    fields = self._read_library_prompt_editor_fields()
    if fields is None:
        return
    raw_name, raw_author, raw_details, raw_system, raw_user, raw_keywords_text = fields

    name = self._sanitize_media_field(raw_name, max_length=300)
    author = self._sanitize_media_field(raw_author, max_length=200)
    details = self._sanitize_note_content(
        raw_details, max_length=source_globals["LIBRARY_PROMPT_TEXT_MAX_CHARS"]
    )
    keywords = self._library_note_keywords_from_input(raw_keywords_text)
    if not name:
        self._update_library_prompt_status_static(
            "Name is required; enter a Prompt name."
        )
        try:
            self.query_one(
                "#mixin-name"
                if self.query(source_globals["PromptMixinEditor"])
                else "#library-prompt-name",
                source_globals["Input"],
            ).focus()
        except (source_globals["NoMatches"], source_globals["QueryError"]):
            pass
        return

    block_state = getattr(self, "_library_prompt_block_state", None)
    prepared_state: PromptBlockEditorState | None = None
    submitted_mixin_draft = None
    artifact_type = self._current_library_prompt_editor_state().artifact_type
    if artifact_type == "mixin":
        mixin_editor = self.query_one(source_globals["PromptMixinEditor"])
        submitted_mixin_draft = mixin_editor.read_draft()
        if not mixin_editor.read_draft()[1].strip():
            self._report_library_mixin_instructions_required()
            return
        try:
            definition = mixin_editor.read_definition()
            draft, save_fields = source_globals["prepare_mixin_artifact_save"](
                definition,
                request_fields={
                    "name": name,
                    "author": author,
                    "details": details,
                    "keywords": keywords,
                    "expected_version": self._library_prompt_version
                    if not is_create
                    else None,
                },
            )
            source_globals["require_artifact_save_supported"](
                draft,
                self._library_prompt_capabilities,
                update_original=not is_create,
                expected_version=self._library_prompt_version
                if not is_create
                else None,
            )
        except ValueError as exc:
            self._update_library_prompt_status_static(str(exc))
            return
    elif block_state is not None:
        artifact_type = target_artifact_type or block_state.artifact_type
        try:
            draft, save_fields, prepared_state = source_globals[
                "prepare_prompt_artifact_save"
            ](
                block_state,
                artifact_type=artifact_type,
                include_recipe_starter_content=(
                    self._library_prompt_include_starter_content
                ),
                request_fields={
                    "name": name,
                    "author": author,
                    "details": details,
                    "keywords": keywords,
                    "expected_version": (
                        self._library_prompt_version if not is_create else None
                    ),
                },
            )
            source_globals["require_artifact_save_supported"](
                draft,
                self._library_prompt_capabilities,
                update_original=not is_create,
                expected_version=(
                    self._library_prompt_version if not is_create else None
                ),
            )
        except ValueError as exc:
            self._update_library_prompt_status_static(str(exc))
            if block_state.issues:
                self._library_prompt_editor_mode = "advanced"
                try:
                    work = self.query_one(
                        "#library-prompt-work-pane",
                        source_globals["LibraryPromptWorkPane"],
                    )
                    await work.set_editor_mode("advanced")
                    work.query_one(
                        "#library-prompt-block-editor",
                        source_globals["PromptBlockEditor"],
                    ).focus_first_error()
                except (source_globals["NoMatches"], source_globals["QueryError"]):
                    pass
            return
    else:
        save_fields = {
            "name": name,
            "author": author,
            "details": details,
            "system_prompt": self._sanitize_note_content(
                raw_system, max_length=source_globals["LIBRARY_PROMPT_TEXT_MAX_CHARS"]
            ),
            "user_prompt": self._sanitize_note_content(
                raw_user, max_length=source_globals["LIBRARY_PROMPT_TEXT_MAX_CHARS"]
            ),
            "keywords": keywords,
        }

    service = getattr(self.app_instance, "prompt_scope_service", None)
    mixin_identity = None
    if artifact_type == "mixin":
        try:
            mixin_identity = service.mixin_source_identity(mode="local")
            original = self._library_prompt_mixin_source_identity
            if not is_create and (
                original is None
                or mixin_identity[0] != original[0]
                or any(a is not b for a, b in zip(mixin_identity[1:], original[1:]))
            ):
                raise ValueError("mixin_source_changed")
        except (ValueError, AttributeError):
            self._update_library_prompt_status_static(
                "Prompt source changed. Reopen this mix-in before saving."
            )
            return

    def source_current():
        if mixin_identity is None:
            return True
        app = self.app_instance
        try:
            current_service = app.prompt_scope_service
        except AttributeError:
            current_service = None
        if current_service is not service:
            return False
        try:
            now = service.mixin_source_identity(mode="local")
            return now[0] == mixin_identity[0] and all(
                a is b for a, b in zip(now[1:], mixin_identity[1:])
            )
        except (ValueError, AttributeError):
            return False

    get_prompt = getattr(service, "get_prompt", None)
    save_prompt = getattr(service, "save_prompt", None)
    if not callable(get_prompt) or not callable(save_prompt):
        return

    if name and name != self._library_prompt_original_name:
        try:
            candidate = await self._run_library_service_call(
                get_prompt,
                mode="local",
                prompt_identifier=name,
                include_deleted=True,
                isolate_in_worker=True,
            )
        except Exception:  # noqa: BLE001 - preserve the existing name-probe fallback.
            candidate = None
        if (
            self._library_prompts_mutation_in_flight
            or selected_prompt_id != self._selected_prompt_id
            or self._library_prompts_view != "editor"
        ):
            return
        candidate_id = (
            candidate.get("local_id")
            if isinstance(candidate, source_globals["Mapping"])
            else None
        )
        if candidate_id is not None and candidate_id != prompt_id:
            if candidate.get("deleted"):
                outcome = source_globals["classify_prompt_save_error"](
                    None, f"Prompt '{name}' exists but is soft-deleted.", None
                )
            else:
                outcome = source_globals["classify_prompt_save_error"](
                    None, f"Prompt '{name}' already exists.", None
                )
            await self._apply_library_prompt_save_outcome(outcome, name=name)
            return

    if not is_create:
        # A not-yet-created prompt has no existing row to have gone
        # stale -- skip the pre-read entirely rather than calling
        # ``get_prompt(prompt_identifier=None)`` (which would only
        # raise, harmlessly swallowed by the ``except`` below, but is
        # wasted work with no real check to perform).
        try:
            fresh = await self._run_library_service_call(
                get_prompt,
                mode="local",
                prompt_identifier=prompt_id,
                include_deleted=True,
                isolate_in_worker=True,
            )
        except Exception:  # noqa: BLE001 - source error bodies remain private.
            # TASK-32801.3: this read is the ONLY staleness check the save
            # path has -- the write seam cannot detect a conflict itself
            # (see this method's own docstring). Swallowing it silently
            # turns "could not tell" into "no conflict", so a transient
            # read failure downgrades a conflicting save into a
            # last-writer-wins overwrite with nothing recorded anywhere.
            source_globals["logger"].warning(
                "Prompt staleness pre-read failed; the save proceeds "
                "without its conflict check (prompt_id={})",
                prompt_id,
            )
            fresh = None
        if (
            self._library_prompts_mutation_in_flight
            or selected_prompt_id != self._selected_prompt_id
            or self._library_prompts_view != "editor"
        ):
            return
        fresh_version = (
            fresh.get("version")
            if isinstance(fresh, source_globals["Mapping"])
            else None
        )
        if (
            fresh_version is not None
            and self._library_prompt_version is not None
            and fresh_version != self._library_prompt_version
        ):
            self._enter_library_prompt_conflict(
                name=raw_name,
                author=raw_author,
                details=raw_details,
                system_prompt=raw_system,
                user_prompt=raw_user,
                keywords_text=raw_keywords_text,
            )
            return

    if not source_current():
        self._update_library_prompt_status_static(
            "Prompt source changed. Reopen this mix-in before saving."
        )
        return
    try:
        result = await self._run_library_service_call(
            save_prompt,
            mode="local",
            prompt_identifier=prompt_id,
            **save_fields,
            isolate_in_worker=True,
        )
    except Exception as exc:  # noqa: BLE001 - classify through existing save recovery.
        source_globals["logger"].opt(exception=True).warning(
            f"Library prompt save failed for {prompt_id!r}."
        )
        if (
            self._library_prompts_mutation_in_flight
            or selected_prompt_id != self._selected_prompt_id
            or self._library_prompts_view != "editor"
        ):
            return
        outcome = source_globals["classify_prompt_save_error"](None, str(exc), exc)
        if outcome == "conflict":
            # A genuine race the pre-checks above could not see (e.g. a
            # second app instance or an external writer landing between
            # this save's pre-read and its real write) -- route into
            # the SAME conflict banner the pre-check staleness path
            # uses above, seeded from the same live (raw, unsanitized)
            # field values, rather than falling through to the generic
            # error status line.
            self._enter_library_prompt_conflict(
                name=raw_name,
                author=raw_author,
                details=raw_details,
                system_prompt=raw_system,
                user_prompt=raw_user,
                keywords_text=raw_keywords_text,
            )
            return
        await self._apply_library_prompt_save_outcome(outcome, name=name)
        return

    if (
        self._library_prompts_mutation_in_flight
        or selected_prompt_id != self._selected_prompt_id
        or self._library_prompts_view != "editor"
    ):
        return

    if not source_current():
        self._update_library_prompt_status_static(
            "Prompt source changed. Reopen this mix-in to check the save result."
        )
        return
    result_id = (
        result.get("local_id")
        if isinstance(result, source_globals["Mapping"])
        else (1 if result else None)
    )
    outcome = source_globals["classify_prompt_save_error"](result_id, "", None)
    if outcome != "ok":
        await self._apply_library_prompt_save_outcome(outcome)
        return

    if save_as_new and artifact_type == "recipe":
        # A Recipe save is an independent library artifact, never the
        # update identity of the active Prompt working copy. Keep the
        # source selection/version/block state and dirty semantics intact.
        self._refresh_local_source_snapshot()
        self._update_library_prompt_status_static("Recipe saved as a new artifact.")
        await self._sync_library_prompt_open_existing_button(show=False)
        return

    from .prompt_save_ack import acknowledge_prompt_save

    await acknowledge_prompt_save(
        self,
        result,
        prompt_id=prompt_id,
        is_create=is_create,
        submitted_fields=fields,
        submitted_state=block_state,
        prepared_state=prepared_state,
        save_fields=save_fields,
        submitted_mixin_draft=submitted_mixin_draft,
    )
