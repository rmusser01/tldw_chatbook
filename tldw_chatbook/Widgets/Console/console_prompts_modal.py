"""Unified Browse/Edit/Improve/Recipe shell for Console prompts."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import DescendantFocus
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Checkbox, Static, TextArea

from tldw_chatbook.Prompt_Management.prompt_artifact_codec import (
    decode_prompt_artifact,
    deserialize_definition,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    BlockArtifactDefinition,
    DecodedPromptArtifact,
    blank_recipe,
    outcome_first_recipe,
)
from tldw_chatbook.Prompt_Management.prompt_legacy_decomposer import (
    decompose_legacy_lanes,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import (
    RECIPE_MAPPED_CONTEXT_BLOCKED_COPY,
    PromptBlockEditor,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor_state import (
    PromptBlockEditorState,
    set_artifact_type,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

from .console_prompts_browse import ConsolePromptsBrowse
from .console_prompt_improve_view import (
    ConsolePromptImprovementContext,
    ConsolePromptImproveView,
    SYSTEM_ANALYSIS_ABSENT_DISCLOSURE,
    SYSTEM_ANALYSIS_ABSENT_TOOLTIP,
    SYSTEM_ANALYSIS_DISCLOSURE,
    SYSTEM_ANALYSIS_LABEL,
    improvement_provider_summary,
)
from .console_composer_bar import ComposerDraftSnapshot
from .console_prompts_state import (
    ConsolePromptsState,
    PromptBrowseResult,
    PromptModalMode,
    PromptSource,
)


_PER_PAGE = 10
_OUTCOME_FIRST_OPTIONAL_BLOCK_IDS = frozenset(
    {
        "role",
        "personality",
        "collaboration-style",
        "success-criteria",
        "stop-rules",
    }
)


@dataclass(frozen=True)
class ConsolePromptsResult:
    """One host-owned, all-or-nothing prompt application request."""

    kind: Literal["apply"]
    composer_snapshot: ComposerDraftSnapshot = field(repr=False)
    user_text: str | None = field(repr=False)
    system_text: str | None = field(repr=False)
    apply_user: bool
    apply_system: bool
    captured_system_fingerprint: str | None = field(repr=False)


@dataclass(frozen=True)
class ConsoleRecipeApplyGuard:
    """Captured Recipe identity paired with a manual editor application."""

    recipe_source: PromptSource | None
    recipe_source_id: str
    recipe_version: int
    recipe_definition: BlockArtifactDefinition = field(repr=False)
    recipe_fingerprint: str = field(repr=False)
    provider_resolution: Any | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ConsoleSavedPromptApplyGuard:
    """Captured saved Prompt identity paired with its editor application."""

    source: PromptSource
    prompt_source_id: str
    prompt_version: int | None
    prompt_definition: BlockArtifactDefinition = field(repr=False)
    prompt_fingerprint: str = field(repr=False)
    record_usage: bool
    provider_resolution: Any | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ConsolePromptsApplyOutcome:
    """Host coordinator result returned before the modal may dismiss."""

    kind: Literal["applied", "persistence_failed", "stale"]
    user_message: str = ""


def _definition_mapping(definition: BlockArtifactDefinition) -> dict[str, Any]:
    return {
        "kind": definition.kind,
        "schema_version": definition.schema_version,
        "lanes": [
            {
                "id": lane.id,
                "blocks": [
                    {
                        "id": block.id,
                        "title": block.title,
                        "syntax": block.syntax,
                        "content": block.content,
                        **(
                            {"xml_tag": block.xml_tag}
                            if block.xml_tag is not None
                            else {}
                        ),
                        **(
                            {"mapping_hint": block.mapping_hint}
                            if block.mapping_hint is not None
                            else {}
                        ),
                    }
                    for block in lane.blocks
                ],
            }
            for lane in definition.lanes
        ],
    }


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _fingerprint_definition(definition: BlockArtifactDefinition) -> str:
    """Import the Task 11 fingerprint lazily to avoid the Console package cycle."""
    from tldw_chatbook.Prompt_Management.prompt_improvement_models import (
        fingerprint_block_definition,
    )

    return fingerprint_block_definition(definition)


def _record_identifier(record: Mapping[str, Any]) -> str:
    value = (
        record.get("source_id")
        or record.get("id")
        or record.get("uuid")
        or record.get("name", "")
    )
    if value in (None, ""):
        raise ValueError("Prompt detail has no source identity.")
    return str(value)


def _record_version(record: Mapping[str, Any]) -> int | None:
    value = record.get("version", record.get("optimistic_version"))
    return value if type(value) is int else None


def _saved_record_identifier(record: Mapping[str, Any]) -> str:
    """Return a durable identity from a normalized save response."""
    value = record.get("source_id") or record.get("id") or record.get("uuid")
    if value in (None, ""):
        raise ValueError("Saved Prompt response has no durable identity.")
    return str(value)


class ConsolePromptsModal(
    SafeModalDismissMixin, ModalScreen[ConsolePromptsResult | None]
):
    """One responsive modal shell with internal prompt-workbench modes."""

    MODES = ("browse", "edit", "improve", "recipe")

    DEFAULT_CSS = """
    ConsolePromptsModal { align: center middle; }
    #console-prompts-modal {
        width: 90%; max-width: 104; height: 90%; max-height: 44;
        min-width: 40; min-height: 18; border: tall $surface-lighten-1;
        background: $panel;
    }
    #console-prompts-header { width: 100%; height: auto; min-height: 2; }
    #console-prompts-title { text-style: bold; color: $text; }
    #console-prompts-location { color: $text-muted; }
    #console-prompts-body { width: 100%; height: 1fr; min-height: 0; }
    #console-prompts-recipe-scroll { width: 100%; height: 1fr; }
    #console-prompts-persistence-retry { display: none; }
    #console-prompts-footer { width: 100%; height: 3; align: right middle; }
    #console-prompts-dirty-guard {
        display: none; width: 100%; height: auto; padding: 1;
        border: round $warning; background: $warning 10%;
    }
    #console-prompts-dirty-guard.visible { display: block; }
    """

    BINDINGS = [("escape", "request_safe_cancel", "Close")]
    SAFE_MODAL_CONTENT = "#console-prompts-modal"

    def __init__(
        self,
        *,
        capabilities: Callable[[str], Any],
        list_page: Callable[[str, int], Any],
        search: Callable[[str, str], Any],
        detail: Callable[[str, str], Any],
        save: Callable[..., Any],
        improve_unavailable_reason: str = "",
        configure_provider: Callable[[], Any] | None = None,
        improvement_context: ConsolePromptImprovementContext | Any | None = None,
        initial_mode: Literal["browse", "improve"] = "browse",
        activate_improvement_context: Callable[[], Any] | None = None,
        capture_manual_resolution: Callable[[], Any] | None = None,
        build_improvement_snapshot: Callable[..., Any] | None = None,
        improve: Callable[[Any], Any] | None = None,
        validate_improvement: Callable[[Any, str], Any] | None = None,
        apply_improvement_result: Callable[[ConsolePromptsResult, Any], Any]
        | None = None,
        retry_improvement_persistence: Callable[[ConsolePromptsResult], Any]
        | None = None,
        open_library_prompt: Callable[[PromptSource, str], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._capabilities = capabilities
        self._list_page = list_page
        self._search = search
        self._detail = detail
        self._save = save
        self._improve_unavailable_reason = improve_unavailable_reason.strip()
        self._configure_provider = configure_provider
        self._improvement_context = improvement_context
        self._activate_improvement_context = activate_improvement_context
        self._capture_manual_resolution = capture_manual_resolution
        self._build_improvement_snapshot = build_improvement_snapshot
        self._improve = improve
        self._validate_improvement = validate_improvement
        self._apply_improvement_result = apply_improvement_result
        self._retry_improvement_persistence = retry_improvement_persistence
        self._open_library_prompt = open_library_prompt
        if initial_mode not in {"browse", "improve"}:
            raise ValueError(f"Unsupported initial Prompt mode: {initial_mode}")
        self.state = ConsolePromptsState(mode_stack=(initial_mode,))
        self.browse_result = PromptBrowseResult(
            source="local", items=(), page=1, total_pages=1, total_items=0
        )
        self._capabilities_by_source: dict[str, object] = {}
        self._debounce_timer: Timer | None = None
        self._editor_state: PromptBlockEditorState | None = None
        self._selected_record: Mapping[str, Any] | None = None
        self._decoded: DecodedPromptArtifact | None = None
        self._compatibility_state = ""
        self._request_counter = 0
        self._active_request_id: str | None = None
        self._apply_in_progress = False
        self._improvement_worker: Any | None = None
        self._activation_counter = 0
        self._active_activation_id: str | None = None
        self._activation_worker: Any | None = None
        self._pending_activation_mode: Literal["auto", "review", "recipe"] | None = None
        recovery = str(getattr(improvement_context, "unavailable_recovery", "provider"))
        self._improve_recovery: Literal["provider", "draft", "reopen"]
        if recovery == "draft":
            self._improve_recovery = "draft"
        elif recovery == "reopen":
            self._improve_recovery = "reopen"
        else:
            self._improve_recovery = "provider"
        self._last_improvement_mode: Literal["auto", "review", "recipe"] | None = None
        self._captured_improvement_request: Any | None = None
        self._pending_persistence_result: ConsolePromptsResult | None = None
        self._recipe_source_id = "builtin:outcome-first"
        self._recipe_source: PromptSource | None = None
        self._recipe_version = 0
        self._recipe_definition: BlockArtifactDefinition | None = None
        self._recipe_source_fingerprint: str | None = None
        self._recipe_selecting = False
        self._saved_recipe_library_target: tuple[PromptSource, str] | None = None
        self._manual_apply_resolution: Any | None = None
        self._include_system_context = bool(
            getattr(improvement_context, "current_system_prompt", "")
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="console-prompts-modal"):
            with Vertical(id="console-prompts-header"):
                yield Static(
                    "Prompt Workbench", id="console-prompts-title", markup=False
                )
                yield Static(
                    self._location_copy(self.state.mode),
                    id="console-prompts-location",
                    markup=False,
                )
            with Vertical(id="console-prompts-body"):
                yield (
                    self._improve_widget()
                    if self.state.mode == "improve"
                    else self._browse_widget()
                )
            with Vertical(id="console-prompts-dirty-guard"):
                yield Static(
                    "Unsaved block changes would be lost. Keep editing, or discard them and go back.",
                    markup=False,
                )
                with Horizontal():
                    yield Button("Keep editing", id="console-prompts-keep-editing")
                    yield Button("Discard changes", id="console-prompts-discard")
            with Horizontal(id="console-prompts-footer"):
                yield Button("Back", id="console-prompts-back")
                yield Button("Close", id="console-prompts-close")

    def _browse_widget(self) -> ConsolePromptsBrowse:
        return ConsolePromptsBrowse(
            source=self.state.source,
            query=self.state.query,
            page=self.state.page,
            improve_unavailable_reason=self._improve_unavailable_reason,
            can_configure_provider=self._can_offer_provider_recovery(),
            manual_improve_available=self._improvement_context is not None,
            id="console-prompts-browse",
        )

    def on_mount(self) -> None:
        self._apply_in_progress = False
        self._set_responsive(self.app.size.width, self.app.size.height)
        if self.state.mode == "browse":
            self.run_worker(
                self.reload_browse(),
                exclusive=False,
                group=f"console-prompts-load-{id(self)}",
            )
        else:
            self._sync_improve_gates()
            if self._improve_unavailable_reason:
                self._set_improvement_status(self._improve_unavailable_reason)
            self.call_after_refresh(
                self._focus_widget,
                self._improve_initial_focus_id(),
            )

    def on_resize(self, event: Any) -> None:
        self._set_responsive(event.size.width, event.size.height)

    def on_unmount(self) -> None:
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
        self._recipe_selecting = False
        self._cancel_improvement_activation()
        if self._active_request_id is not None:
            self._active_request_id = None
            worker = self._improvement_worker
            if worker is not None:
                worker.cancel()
        self.call_after_refresh(self._restore_composer_focus)

    def _set_responsive(self, width: int, height: int) -> None:
        narrow = width < 96 or height < 30
        self.set_class(narrow, "-narrow")

    def _restore_composer_focus(self) -> None:
        for screen in reversed(tuple(self.app.screen_stack)):
            try:
                composer = screen.query_one("#console-native-composer")
            except NoMatches:
                continue
            composer.can_focus = True
            composer.focus()
            return

    def _remember_current_focus(self) -> None:
        focused = self.app.focused
        widget_id = getattr(focused, "id", None)
        if widget_id:
            self.state = self.state.remember_focus(self.state.mode, widget_id)

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        widget_id = event.widget.id
        if not widget_id:
            return
        try:
            body = self.query_one("#console-prompts-body", Vertical)
        except NoMatches:
            return
        if body in event.widget.ancestors:
            self.state = self.state.remember_focus(self.state.mode, widget_id)

    def _focus_widget(self, widget_id: str | None) -> None:
        if not widget_id:
            return
        try:
            widget = self.query_one(f"#{widget_id}")
        except NoMatches:
            return
        widget.focus()
        widget.scroll_visible(animate=False, immediate=True)

    async def enter_mode(
        self, mode: PromptModalMode, *, focus_id: str | None = None
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported prompt modal mode: {mode}")
        if self._recipe_selecting and self.state.mode == "browse" and mode != "browse":
            self._recipe_selecting = False
        self._remember_current_focus()
        self.state = self.state.enter_mode(mode)
        await self._mount_mode(mode)
        self._reset_persistence_retry()
        self.call_after_refresh(
            self._focus_widget,
            focus_id or self.state.focus_for(mode),
        )

    async def _back_internal(self, *, discard: bool = False) -> None:
        if self._apply_in_progress:
            return
        self._cancel_improvement_activation()
        self._reset_persistence_retry()
        if self._active_request_id is not None:
            self._cancel_improvement()
            return
        if self._recipe_selecting and self.state.mode == "browse":
            self._recipe_selecting = False
        if self.state.mode in {"edit", "recipe"} and self.state.dirty and not discard:
            self._show_dirty_guard()
            return
        if len(self.state.mode_stack) == 1:
            self.dismiss(None)
            return
        self._remember_current_focus()
        previous = self.state.mode_stack[-2]
        focus_id = self.state.focus_for(previous)
        self.state = self.state.go_back()
        await self._mount_mode(previous)
        self.call_after_refresh(self._focus_widget, focus_id)

    async def _mount_mode(self, mode: PromptModalMode) -> None:
        body = self.query_one("#console-prompts-body", Vertical)
        await body.remove_children()
        self.query_one("#console-prompts-location", Static).update(
            self._location_copy(mode)
        )
        if mode == "browse":
            browse = self._browse_widget()
            await body.mount(browse)
            await browse.show_result(self.browse_result, query=self.state.query)
        elif mode == "edit":
            if self._editor_state is not None:
                if self._decoded is not None and self._decoded.compatibility_stale:
                    await body.mount(
                        Static(
                            "Saved compiled text differs from the blocks. The block definition is authoritative; Saving repairs the compiled System and User fields.",
                            id="console-prompts-compatibility-stale",
                            markup=False,
                        )
                    )
                await body.mount(
                    Static(
                        "",
                        id="console-prompts-improvement-status",
                        markup=False,
                    )
                )
                editor = PromptBlockEditor(
                    self._editor_state,
                    can_update_original=self._can_update_original(),
                    show_back=False,
                    id="console-prompts-editor",
                )
                await body.mount(editor)
                self._sync_editor_host_gates(editor)
            else:
                await self._mount_compatibility(body)
        elif mode == "improve":
            await body.mount(self._improve_widget())
            self._sync_improve_gates()
        else:
            await self._mount_recipe_choices(body)

    def _location_copy(self, mode: PromptModalMode) -> str:
        parts = ["Prompts", mode.title()]
        if mode == "edit" and self.state.selected_source:
            parts.append(self.state.selected_source.title())
            selected_name = str((self._selected_record or {}).get("name") or "").strip()
            if selected_name:
                parts.append(selected_name)
            if self._decoded is not None:
                parts.append(self._decoded.artifact_type.title())
            if self.state.working_copy_unsaved:
                parts.append("Unsaved working copy")
        return " / ".join(parts)

    def _can_offer_provider_recovery(self) -> bool:
        """Return whether the current blocker is repairable in provider settings."""

        return bool(
            self._improve_unavailable_reason
            and self._improve_recovery == "provider"
            and self._configure_provider is not None
        )

    def _improve_widget(self) -> ConsolePromptImproveView | Vertical:
        if self._improvement_context is not None:
            return ConsolePromptImproveView(
                self._improvement_context,
                model_unavailable_reason=self._improve_unavailable_reason,
                show_configure_provider=self._can_offer_provider_recovery(),
                include_system_context=self._include_system_context,
                id="console-prompts-improve-mode",
            )
        container = Vertical(id="console-prompts-improve-mode")
        container.compose_add_child(
            Static(
                "Choose how the current unsent message should be improved. No model call starts until you choose a path.",
                markup=False,
            )
        )
        for label, button_id in (
            ("Replace draft automatically", "console-prompts-auto-improve"),
            (
                "Analyze and user review (Recommended)",
                "console-prompts-review-improve",
            ),
            (
                "Build a reusable prompt",
                "console-prompts-structured-recipe",
            ),
        ):
            model_action = button_id != "console-prompts-structured-recipe"
            disabled = model_action and bool(self._improve_unavailable_reason)
            button = Button(label, id=button_id, disabled=disabled)
            if disabled:
                button.tooltip = self._improve_unavailable_reason
            container.compose_add_child(button)
        if self._can_offer_provider_recovery():
            container.compose_add_child(
                Button(
                    "Configure provider / model",
                    id="console-prompts-configure-provider",
                )
            )
        return container

    async def _mount_recipe_choices(self, body: Vertical | None = None) -> None:
        if body is None:
            body = self.query_one("#console-prompts-body", Vertical)
            await body.remove_children()
        await body.mount(
            VerticalScroll(
                Static(
                    "Choose a starting point. Saved formats remain Recipes in Library > Prompts.",
                    markup=False,
                ),
                Button(
                    "Outcome-first",
                    id="console-prompts-recipe-outcome-first",
                ),
                Static(
                    "Outcome-first starts with Goal, context and evidence, constraints, and output; reveal optional guidance when useful.",
                    id="console-prompts-recipe-outcome-description",
                    markup=False,
                ),
                Button("Saved Recipe", id="console-prompts-recipe-saved"),
                Static(
                    "Saved Recipe reuses a format from Library > Prompts.",
                    id="console-prompts-recipe-saved-description",
                    markup=False,
                ),
                Button("Blank", id="console-prompts-recipe-blank"),
                Static(
                    "Blank starts with empty System and User lanes for your own blocks.",
                    id="console-prompts-recipe-blank-description",
                    markup=False,
                ),
                Static("", id="console-prompts-improvement-status", markup=False),
                id="console-prompts-recipe-mode",
            )
        )

    def _can_update_original(self) -> bool:
        capabilities = self.state.selected_capabilities
        return bool(
            not self.state.working_copy_unsaved
            and self._supports_structured_save("prompt")
            and capabilities is not None
            and getattr(capabilities, "conditional_update", False)
            and self.state.selected_version is not None
        )

    def _supports_structured_save(self, artifact_type: str) -> bool:
        source = self.state.selected_source or self.state.source
        capabilities = self.state.selected_capabilities or self._capabilities_by_source.get(
            source
        )
        if capabilities is None:
            return False
        kind = "block_recipe" if artifact_type == "recipe" else "block_prompt"
        return bool(
            artifact_type in getattr(capabilities, "artifact_types", frozenset())
            and (2, kind) in getattr(capabilities, "structured_kinds", frozenset())
        )

    def _sync_editor_host_gates(self, editor: PromptBlockEditor | None = None) -> None:
        if editor is None:
            try:
                editor = self.query_one(PromptBlockEditor)
            except NoMatches:
                return
        editor.set_update_original_available(
            self._can_update_original(),
            unavailable_reason=(
                "Update unavailable — this source does not support conditional updates "
                "for this version; save as new."
            ),
        )
        source = self.state.selected_source or self.state.source
        prompt_supported = self._supports_structured_save("prompt")
        recipe_supported = self._supports_structured_save("recipe")
        editor.set_save_capabilities(
            prompt=prompt_supported,
            recipe=recipe_supported,
            prompt_unavailable_reason=(
                "" if prompt_supported else f"{source} does not support block_prompt."
            ),
            recipe_unavailable_reason=(
                "" if recipe_supported else f"{source} does not support block_recipe."
            ),
        )
        can_apply_saved_prompt = bool(
            self._improvement_context is not None
            and self._apply_improvement_result is not None
            and self.state.mode == "edit"
            and not self.state.working_copy_unsaved
            and self.state.selected_identity
            and self.state.selected_source
            and self._decoded is not None
            and self._decoded.artifact_type == "prompt"
            and self._decoded.definition is not None
        )
        if can_apply_saved_prompt:
            return
        apply_button = editor.query_one("#prompt-editor-apply", Button)
        apply_button.disabled = True
        apply_button.tooltip = "Applying to the composer is unavailable in this stage; save the Prompt instead."
        apply_reason = editor.query_one("#prompt-editor-apply-reason", Static)
        apply_reason.update(
            "Apply unavailable — save this Prompt. System changes only on Apply in "
            "this active session."
        )
        apply_reason.add_class("blocked")

    async def _mount_compatibility(self, body: Vertical) -> None:
        record = self._selected_record or {}
        system = str(record.get("system_prompt") or "")
        user = str(record.get("user_prompt") or "")
        await body.mount(
            Static(
                (
                    f"Compatibility view · {self._compatibility_state}. "
                    "This structure cannot be edited losslessly here; inspect the compiled lanes or convert a new Prompt."
                ),
                id="console-prompts-compatibility",
                markup=False,
            ),
            Static("System compatibility text", markup=False),
            TextArea(system, read_only=True, id="console-prompts-compat-system"),
            Static("User compatibility text", markup=False),
            TextArea(user, read_only=True, id="console-prompts-compat-user"),
        )
        convert = Button(
            "Convert and save as new",
            id="console-prompts-convert",
            disabled=not bool(system or user),
        )
        if convert.disabled:
            convert.tooltip = "Conversion unavailable — this artifact has no compatible System or User text."
        await body.mount(convert)

    async def set_query(self, query: str) -> None:
        self.state = self.state.with_query(query).begin_search()
        await self.reload_browse(token_already_started=True)

    async def switch_source(self, source: PromptSource) -> None:
        if source == self.state.source:
            return
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
            self._debounce_timer = None
        self.state = self.state.with_source(source)
        await self._clear_cross_source_results(source)
        await self.reload_browse(token_already_started=True)

    async def _clear_cross_source_results(self, source: PromptSource) -> None:
        if self.browse_result.source == source:
            return
        self.browse_result = PromptBrowseResult(
            source=source,
            items=(),
            page=1,
            total_pages=1,
            total_items=0,
        )
        try:
            browse = self.query_one(ConsolePromptsBrowse)
        except NoMatches:
            return
        await browse.show_result(self.browse_result, query=self.state.query)

    async def reload_browse(self, *, token_already_started: bool = False) -> None:
        if not token_already_started:
            self.state = self.state.begin_search()
        token = self.state.search_token
        source = self.state.source
        query = self.state.query.strip()
        try:
            browse = self.query_one(ConsolePromptsBrowse)
            await self._clear_cross_source_results(source)
            browse.show_loading(source=source, query=query)
        except NoMatches:
            return
        try:
            capabilities = await _maybe_await(self._capabilities(source))
            if not query:
                raw = await _maybe_await(self._list_page(source, self.state.page))
                result = self._normalize_list_result(raw, source, self.state.page)
            else:
                raw = await _maybe_await(self._search(source, query))
                result = self._normalize_search_result(raw, source)
        except Exception as exc:
            if not self.state.accepts(token, source):
                return
            self._show_browse_error(exc, source=source, query=query)
            return
        if not self.state.accepts(token, source) or result.source != source:
            return
        self._capabilities_by_source[source] = capabilities
        self.browse_result = result
        await browse.show_result(result, query=query)

    @staticmethod
    def _normalize_list_result(
        raw: Any, source: PromptSource, requested_page: int
    ) -> PromptBrowseResult:
        if isinstance(raw, PromptBrowseResult):
            return raw
        data = raw if isinstance(raw, Mapping) else {}
        items_value = data.get("items", ())
        items = tuple(item for item in items_value if isinstance(item, Mapping))
        page = data.get("page")
        page = page if type(page) is int and page > 0 else requested_page
        total = data.get("total_items", data.get("total", len(items)))
        total = total if type(total) is int and total >= 0 else len(items)
        total_pages = data.get("total_pages")
        if type(total_pages) is not int or total_pages < 1:
            per_page = data.get("per_page", _PER_PAGE)
            per_page = per_page if type(per_page) is int and per_page > 0 else _PER_PAGE
            total_pages = max(1, math.ceil(total / per_page))
        return PromptBrowseResult(
            source=source,
            items=items,
            page=page,
            total_pages=total_pages,
            total_items=total,
        )

    @staticmethod
    def _normalize_search_result(raw: Any, source: PromptSource) -> PromptBrowseResult:
        if isinstance(raw, PromptBrowseResult):
            return raw
        if isinstance(raw, Mapping):
            items_value = raw.get("items", ())
        else:
            items_value = raw or ()
        items = tuple(item for item in items_value if isinstance(item, Mapping))
        return PromptBrowseResult(
            source=source,
            items=items,
            page=1,
            total_pages=1,
            total_items=len(items),
        )

    def _show_browse_error(
        self, exc: Exception, *, source: PromptSource, query: str
    ) -> None:
        browse = self.query_one(ConsolePromptsBrowse)
        if query:
            message = (
                f"Search failed for {source.title()} — results were not changed. "
                "Retry, change the query, or switch source."
            )
        elif "unavailable" in str(exc).lower() or isinstance(exc, ValueError):
            message = (
                f"{source.title()} Prompt source is unavailable — its Library cannot be shown. "
                "Retry or switch source."
            )
        else:
            message = (
                f"{source.title()} Prompt Library failed to load — no items were replaced. "
                "Retry or switch source."
            )
        browse.show_status(message, retry=True)

    async def open_artifact(self, identifier: str) -> None:
        source = self.state.source
        self.state = self.state.begin_detail(identifier)
        detail_token = self.state.detail_token
        browse = self.query_one(ConsolePromptsBrowse)
        browse.show_status(
            f"Loading latest {source.title()} detail before editing…",
            retry=False,
        )
        try:
            record_value = await _maybe_await(self._detail(source, identifier))
            if not isinstance(record_value, Mapping):
                raise KeyError(identifier)
            record = dict(record_value)
            identity = _record_identifier(record)
            capabilities = self._capabilities_by_source.get(source)
            if capabilities is None:
                capabilities = await _maybe_await(self._capabilities(source))
        except Exception:
            if not self.state.accepts_detail(detail_token, source, identifier):
                return
            browse.show_status(
                "The selected artifact was changed or deleted before its latest detail could be loaded — Retry the Library or choose another item.",
                retry=True,
            )
            return

        if not self.state.accepts_detail(detail_token, source, identifier):
            return

        try:
            decoded = decode_prompt_artifact(record)
        except (TypeError, ValueError):
            decoded = DecodedPromptArtifact(
                state="malformed",
                artifact_type="prompt",
                definition=None,
                raw_definition=deserialize_definition(record.get("prompt_definition")),
                compiled_system=str(record.get("system_prompt") or ""),
                compiled_user=str(record.get("user_prompt") or ""),
                compatibility_stale=False,
            )

        if self._recipe_selecting:
            if (
                decoded.state != "supported_v2"
                or decoded.artifact_type != "recipe"
                or decoded.definition is None
            ):
                browse.show_status(
                    "Choose a supported schema-v2 Recipe for this structured flow.",
                    retry=False,
                )
                return
            await self._capture_manual_apply_target()
            self._recipe_selecting = False
            self._recipe_source = source
            self._recipe_source_id = identity
            self._recipe_version = _record_version(record) or 0
            self._recipe_definition = decoded.definition
            self._recipe_source_fingerprint = _fingerprint_definition(
                decoded.definition
            )
            self.state = self.state.go_back()
            await self._mount_recipe_editor(decoded.definition)
            return

        self._selected_record = record
        self._decoded = decoded
        self.state = self.state.select(
            identity=identity,
            version=_record_version(record),
            source=source,
            capabilities=capabilities,
        )
        if decoded.state == "supported_v2" and decoded.definition is not None:
            definition = decoded.definition
            unsaved = decoded.artifact_type == "recipe"
            if unsaved:
                definition = replace(definition, kind="block_prompt")
            else:
                await self._capture_manual_apply_target()
            self._editor_state = PromptBlockEditorState.from_definition(
                artifact_type="prompt", definition=definition
            )
            self.state = self.state.as_unsaved_copy(unsaved)
            await self.enter_mode("edit")
            return
        if decoded.state == "legacy" and decoded.artifact_type == "prompt":
            decomposition = decompose_legacy_lanes(
                decoded.compiled_system, decoded.compiled_user
            )
            self._editor_state = PromptBlockEditorState.from_definition(
                artifact_type="prompt",
                definition=decomposition.definition,
                system_origin=decomposition.system_origin,
                user_origin=decomposition.user_origin,
            )
            self.state = self.state.as_unsaved_copy(False)
            await self.enter_mode("edit")
            return

        self._editor_state = None
        self._compatibility_state = decoded.state.replace("_", " ")
        await self.enter_mode("edit")

    async def _capture_manual_apply_target(self) -> None:
        """Pin the provider resolution used to guard a later manual Apply."""

        self._manual_apply_resolution = None
        if self._capture_manual_resolution is None:
            return
        try:
            self._manual_apply_resolution = await _maybe_await(
                self._capture_manual_resolution()
            )
        except Exception:
            self._manual_apply_resolution = None

    def mark_dirty(self) -> None:
        self.state = self.state.with_dirty(True)

    def _context_has_improvable_text(self) -> bool:
        context = self._improvement_context
        projection = getattr(context, "current_user_projection", None)
        text = str(getattr(projection, "text", "") or "")
        for token in tuple(getattr(projection, "placeholder_ids", ()) or ()):
            text = text.replace(str(token), "")
        return bool(text.strip())

    def _sync_improve_gates(self, *, active: bool = False) -> None:
        has_text = self._context_has_improvable_text()
        for selector in (
            "#console-prompts-auto-improve",
            "#console-prompts-review-improve",
            "#console-prompts-recipe-fill",
        ):
            try:
                button = self.query_one(selector, Button)
            except NoMatches:
                continue
            button.disabled = (
                active or bool(self._improve_unavailable_reason) or not has_text
            )
            if active:
                button.tooltip = (
                    "Wait for the current improvement to finish or cancel it."
                )
            elif not has_text:
                button.tooltip = "Add text to the unsent message before improving it."
            elif self._improve_unavailable_reason:
                button.tooltip = self._improve_unavailable_reason
            else:
                button.tooltip = None
        try:
            structured = self.query_one("#console-prompts-structured-recipe", Button)
        except NoMatches:
            return
        structured.disabled = active
        structured.tooltip = (
            "Wait for the current provider resolution to finish." if active else None
        )

    def _set_improvement_status(self, message: str) -> None:
        try:
            self.query_one("#console-prompts-improvement-status", Static).update(
                message
            )
        except NoMatches:
            return

    def _set_improvement_action_visible(self, selector: str, visible: bool) -> None:
        try:
            self.query_one(selector, Button).display = visible
        except NoMatches:
            return

    def _set_persistence_retry_visible(
        self,
        visible: bool,
        *,
        focus: bool = False,
    ) -> None:
        try:
            retry = self.query_one("#console-prompts-persistence-retry", Button)
        except NoMatches:
            return
        retry.display = visible
        retry.disabled = not visible
        retry.can_focus = visible
        if visible and focus:
            self.call_after_refresh(
                self._focus_widget,
                "console-prompts-persistence-retry",
            )

    def _reset_persistence_retry(self) -> None:
        self._pending_persistence_result = None
        self._set_persistence_retry_visible(False)

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"prompt-improvement-{self._request_counter}"

    def _next_activation_id(self) -> str:
        self._activation_counter += 1
        return f"prompt-improvement-activation-{self._activation_counter}"

    def _set_activation_busy(self, active: bool) -> None:
        try:
            browse = self.query_one(ConsolePromptsBrowse)
            improve = browse.query_one("#console-prompts-improve", Button)
        except NoMatches:
            return
        improve.disabled = active or bool(
            self._improve_unavailable_reason and self._improvement_context is None
        )
        if active:
            improve.tooltip = "Resolving the current provider, model, and endpoint."
            browse.show_status(
                "Resolving current provider, model, and endpoint…",
                retry=False,
            )
        else:
            improve.tooltip = self._improve_unavailable_reason or None

    def _begin_improvement_activation(
        self, mode: Literal["auto", "review", "recipe"]
    ) -> None:
        if self._active_activation_id is not None:
            return
        activation_id = self._next_activation_id()
        self._active_activation_id = activation_id
        self._pending_activation_mode = mode
        if self.state.mode in {"improve", "recipe"}:
            self._sync_improve_gates(active=True)
            self._set_improvement_status(
                "Resolving current provider, model, and endpoint…"
            )
        else:
            self._set_activation_busy(True)
        self._activation_worker = self.run_worker(
            self._run_improvement_activation(activation_id),
            exclusive=True,
            group=f"console-prompt-improvement-activation-{id(self)}",
        )

    def _improve_initial_focus_id(self) -> str:
        if self._can_offer_provider_recovery():
            return "console-prompts-configure-provider"
        if self._improve_unavailable_reason:
            return "console-prompts-structured-recipe"
        return "console-prompts-review-improve"

    def _cancel_improvement_activation(self) -> bool:
        if self._active_activation_id is None:
            return False
        self._active_activation_id = None
        self._pending_activation_mode = None
        worker = self._activation_worker
        self._activation_worker = None
        if worker is not None:
            worker.cancel()
        return True

    async def _run_improvement_activation(self, activation_id: str) -> None:
        try:
            if self._activate_improvement_context is not None:
                try:
                    context = await _maybe_await(self._activate_improvement_context())
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    if self._active_activation_id != activation_id:
                        return
                    if isinstance(exc, ValueError) and str(exc).strip():
                        self._improve_unavailable_reason = str(exc).strip()
                        self._improve_recovery = "reopen"
                    else:
                        self._improve_unavailable_reason = (
                            "Prompt improvement could not resolve the current provider "
                            "target. Review Console provider settings and reopen Improve."
                        )
                        self._improve_recovery = "provider"
                else:
                    if self._active_activation_id != activation_id:
                        return
                    if context is not None:
                        self._improvement_context = context
                        pinned_resolution = getattr(
                            context,
                            "pinned_resolution",
                            None,
                        )
                        if pinned_resolution is not None:
                            self._manual_apply_resolution = pinned_resolution
                        self._improve_unavailable_reason = str(
                            getattr(context, "model_unavailable_reason", "") or ""
                        ).strip()
                        recovery = str(
                            getattr(context, "unavailable_recovery", "provider")
                        )
                        if recovery == "reopen":
                            self._improve_recovery = "reopen"
                        elif recovery == "draft":
                            self._improve_recovery = "draft"
                        else:
                            self._improve_recovery = "provider"
            if self._active_activation_id != activation_id:
                return
            pending_mode = self._pending_activation_mode
            if self.state.mode == "improve" and self._improve_unavailable_reason:
                await self._mount_mode("improve")
            else:
                try:
                    self.query_one("#console-prompts-provider-summary", Static).update(
                        improvement_provider_summary(self._improvement_context)
                    )
                except NoMatches:
                    pass
            if self._improve_unavailable_reason:
                self._sync_improve_gates()
                self._set_improvement_status(self._improve_unavailable_reason)
                if self.state.mode == "improve":
                    self.call_after_refresh(
                        self._focus_widget,
                        self._improve_initial_focus_id(),
                    )
                return
            if pending_mode is not None:
                self._begin_improvement(pending_mode)
        finally:
            if self._active_activation_id == activation_id:
                self._active_activation_id = None
                self._activation_worker = None
                self._pending_activation_mode = None
                self._set_activation_busy(False)

    def _begin_improvement(self, mode: Literal["auto", "review", "recipe"]) -> None:
        if self._active_request_id is not None:
            return
        if mode != "recipe" and not self._context_has_improvable_text():
            self._set_improvement_status(
                "Add text to the unsent message before improving it."
            )
            return
        if self._improve is None or self._build_improvement_snapshot is None:
            self._set_improvement_status(
                "Prompt improvement is unavailable until a provider and model are ready."
            )
            return
        self._last_improvement_mode = mode
        request_id = self._next_request_id()
        self._active_request_id = request_id
        self._sync_improve_gates(active=True)
        self._improvement_worker = self.run_worker(
            self._run_improvement(mode, request_id),
            exclusive=True,
            group="console-prompt-improvement",
        )

    async def _run_improvement(
        self,
        mode: Literal["auto", "review", "recipe"],
        request_id: str,
    ) -> None:
        self._set_improvement_status("Improving…")
        self._set_improvement_action_visible(
            "#console-prompts-improvement-cancel", True
        )
        self._set_improvement_action_visible(
            "#console-prompts-improvement-retry", False
        )
        try:
            include_system = False
            try:
                include_system = self.query_one(
                    "#console-prompts-include-system", Checkbox
                ).value
            except NoMatches:
                include_system = self._include_system_context
            recipe_definition = None
            if mode == "recipe":
                editor_state = self._editor_state
                if editor_state is None:
                    raise ValueError("Choose or build a Recipe before using AI fill.")
                recipe_definition = replace(
                    editor_state.definition,
                    kind="block_recipe",
                )
                PromptBlockEditorState.from_definition(
                    artifact_type="recipe",
                    definition=recipe_definition,
                )
            captured = await _maybe_await(
                self._build_improvement_snapshot(
                    mode=mode,
                    request_id=request_id,
                    include_system=bool(include_system),
                    recipe_source=(
                        self._recipe_source if recipe_definition is not None else None
                    ),
                    recipe_source_id=(
                        self._recipe_source_id
                        if recipe_definition is not None
                        else None
                    ),
                    recipe_version=(
                        self._recipe_version if recipe_definition is not None else None
                    ),
                    recipe_definition=recipe_definition,
                    recipe_fingerprint=(
                        _fingerprint_definition(recipe_definition)
                        if recipe_definition is not None
                        else None
                    ),
                )
            )
            self._captured_improvement_request = captured
            outcome = await _maybe_await(self._improve(captured))
        except ValueError as exc:
            if self._active_request_id == request_id:
                self._active_request_id = None
                self._set_improvement_status(str(exc))
                self._set_improvement_action_visible(
                    "#console-prompts-improvement-retry", True
                )
                self._sync_improve_gates()
            return
        except Exception:
            if self._active_request_id == request_id:
                self._active_request_id = None
                self._set_improvement_status(
                    "Prompt improvement could not start. Check the provider and retry."
                )
                self._set_improvement_action_visible(
                    "#console-prompts-improvement-retry", True
                )
                self._sync_improve_gates()
            return
        finally:
            if self._active_request_id in {None, request_id}:
                self._set_improvement_action_visible(
                    "#console-prompts-improvement-cancel", False
                )
        if self._active_request_id != request_id:
            return
        if getattr(outcome, "request_id", None) != request_id:
            self._active_request_id = None
            self._sync_improve_gates()
            self._set_improvement_status(
                "The improvement result is stale. Capture the prompt again and retry."
            )
            return
        self._active_request_id = None
        await self._handle_improvement_outcome(outcome, captured)
        self._sync_improve_gates()

    async def _handle_improvement_outcome(self, outcome: Any, captured: Any) -> None:
        kind = str(getattr(outcome, "kind", "provider_error"))
        if kind == "no_change":
            self._set_improvement_status("Prompt already looks good")
            return
        if kind == "success":
            if getattr(captured, "mode", "") == "auto":
                await self._coordinate_apply(
                    self._result_for(
                        user_text=getattr(outcome, "rewritten_prompt", None),
                        system_text=None,
                        apply_user=True,
                        apply_system=False,
                        captured=captured,
                    ),
                    captured,
                )
                return
            if getattr(captured, "mode", "") == "review":
                await self._mount_review(str(outcome.rewritten_prompt or ""))
                return
            definition = getattr(outcome, "filled_definition", None)
            if isinstance(definition, BlockArtifactDefinition):
                if definition.kind != "block_prompt":
                    self._set_improvement_status(
                        "Recipe fill returned an incompatible artifact. Retry the fill."
                    )
                    return
                await self._mount_recipe_editor(
                    definition,
                    artifact_type="prompt",
                )
                return
        if (
            kind == "preservation_veto"
            and getattr(outcome, "rewritten_prompt", None) is not None
        ):
            await self._mount_review(str(outcome.rewritten_prompt))
            self._set_improvement_status("Review required before applying")
            return
        self._set_improvement_status(
            str(getattr(outcome, "user_message", "") or "Prompt improvement failed.")
        )
        self._set_improvement_action_visible(
            "#console-prompts-improvement-retry",
            kind in {"empty", "provider_error", "malformed", "context_limit"},
        )

    def _result_for(
        self,
        *,
        user_text: str | None,
        system_text: str | None,
        apply_user: bool,
        apply_system: bool,
        captured: Any,
    ) -> ConsolePromptsResult:
        context = self._improvement_context
        snapshot = getattr(captured, "composer_snapshot", None) or getattr(
            context, "composer_snapshot"
        )
        return ConsolePromptsResult(
            kind="apply",
            composer_snapshot=snapshot,
            user_text=user_text if apply_user else None,
            system_text=system_text if apply_system else None,
            apply_user=apply_user,
            apply_system=apply_system,
            captured_system_fingerprint=(
                getattr(context, "current_system_fingerprint", None)
                if apply_system
                else None
            ),
        )

    async def _coordinate_apply(
        self,
        result: ConsolePromptsResult,
        captured: Any,
        *,
        _generation: int | None = None,
    ) -> None:
        generation = _generation
        if generation is None:
            generation = self._claim_apply_transaction()
            if generation is None:
                return
        if not self._apply_presentation_is_current(generation):
            return
        self._reset_persistence_retry()
        if self._apply_improvement_result is None:
            self.dismiss_safe_once(result)
            self._set_apply_in_progress(False)
            return
        try:
            try:
                outcome = await _maybe_await(
                    self._apply_improvement_result(result, captured)
                )
            except Exception:
                if self._apply_presentation_is_current(generation):
                    self._set_improvement_status(
                        "The prompt changed while this result was open. Capture it again and retry."
                    )
                return
            if not self._apply_presentation_is_current(generation):
                return
            kind = str(getattr(outcome, "kind", "applied"))
            if kind == "applied":
                if not self.dismiss_safe_once(result):
                    self._set_improvement_status("Applied to the Console.")
                return
            if kind == "persistence_failed":
                self._pending_persistence_result = result
                self._set_improvement_status(
                    "Applied to this session, but could not save to the conversation."
                )
                self._set_persistence_retry_visible(
                    True,
                    focus=True,
                )
                return
            if (
                kind == "stale"
                and getattr(captured, "mode", "") in {"auto", "review"}
                and result.apply_user
                and result.user_text is not None
            ):
                await self._mount_review(result.user_text)
            self._set_improvement_status(
                str(
                    getattr(outcome, "user_message", "")
                    or "The live Console state changed. Capture the prompt again and retry."
                )
            )
        finally:
            if self._safe_mount_generation == generation:
                self._set_apply_in_progress(False)

    def _apply_presentation_is_current(self, generation: int) -> bool:
        return self.is_mounted and self._safe_mount_generation == generation

    def _claim_apply_transaction(self) -> int | None:
        if self._apply_in_progress or not self.is_mounted:
            return None
        self._set_apply_in_progress(True)
        return self._safe_mount_generation

    def _set_apply_in_progress(self, active: bool) -> None:
        self._apply_in_progress = active
        for selector in (
            "#console-prompts-back",
            "#console-prompts-close",
            "#console-prompts-review-apply",
            "#prompt-editor-apply",
        ):
            try:
                self.query_one(selector, Button).disabled = active
            except NoMatches:
                continue
        if not self.is_mounted:
            return
        try:
            status = self.query_one("#console-prompts-improvement-status", Static)
        except NoMatches:
            return
        status.can_focus = active
        if active:
            status.update("Applying changes to the Console…")
            status.focus()
        else:
            self._sync_editor_host_gates()

    async def _mount_review(self, candidate: str) -> None:
        body = self.query_one("#console-prompts-body", Vertical)
        await body.remove_children()
        await body.mount(
            Vertical(
                Static(
                    "Review and edit the proposed User message before applying it.",
                    markup=False,
                ),
                TextArea(candidate, id="console-prompts-review-user"),
                Static("", id="console-prompts-improvement-status", markup=False),
                Horizontal(
                    Button("Apply", id="console-prompts-review-apply"),
                    Button("Retry", id="console-prompts-improvement-retry"),
                ),
                id="console-prompts-review-mode",
            )
        )

    async def _apply_review_candidate(self, generation: int) -> None:
        if not self._apply_presentation_is_current(generation):
            return
        captured = self._captured_improvement_request
        if captured is None:
            self._set_improvement_status("The captured request is no longer available.")
            self._set_apply_in_progress(False)
            return
        candidate = self.query_one("#console-prompts-review-user", TextArea).text
        try:
            if self._validate_improvement is not None:
                await _maybe_await(self._validate_improvement(captured, candidate))
        except Exception:
            if not self._apply_presentation_is_current(generation):
                return
            self._set_improvement_status(
                "Protected prompt material changed. Restore the protected placeholders before applying."
            )
            self._set_apply_in_progress(False)
            return
        if not self._apply_presentation_is_current(generation):
            return
        await self._coordinate_apply(
            self._result_for(
                user_text=candidate,
                system_text=None,
                apply_user=True,
                apply_system=False,
                captured=captured,
            ),
            captured,
            _generation=generation,
        )

    async def _mount_recipe_editor(
        self,
        definition: BlockArtifactDefinition,
        *,
        artifact_type: Literal["prompt", "recipe"] = "recipe",
        initially_hidden_block_ids: frozenset[str] = frozenset(),
    ) -> None:
        if artifact_type == "recipe":
            self._recipe_definition = definition
        self._editor_state = PromptBlockEditorState.from_definition(
            artifact_type=artifact_type,
            definition=definition,
        )
        if artifact_type == "prompt":
            self.state = self.state.as_unsaved_copy(True).with_dirty(True)
        body = self.query_one("#console-prompts-body", Vertical)
        await body.remove_children()
        self._reset_persistence_retry()
        heading = (
            "Filled Prompt review"
            if artifact_type == "prompt"
            else "Structured Recipe working copy"
        )
        retry = Button(
            "Retry save",
            id="console-prompts-persistence-retry",
            disabled=True,
        )
        retry.display = False
        retry.can_focus = False
        widgets: list[Any] = [
            Static(heading, markup=False),
            Static(
                improvement_provider_summary(self._improvement_context),
                id="console-prompts-provider-summary",
                markup=False,
            ),
        ]
        if artifact_type == "recipe":
            fill_disabled = (
                bool(self._improve_unavailable_reason)
                or not self._context_has_improvable_text()
            )
            fill = Button(
                "Fill with AI",
                id="console-prompts-recipe-fill",
                disabled=fill_disabled,
            )
            if fill_disabled:
                fill.tooltip = (
                    self._improve_unavailable_reason
                    or "Add text to the unsent message before using AI fill."
                )
            has_system = bool(
                getattr(
                    self._improvement_context,
                    "current_system_prompt",
                    "",
                )
            )
            analysis_context = Checkbox(
                SYSTEM_ANALYSIS_LABEL,
                value=self._include_system_context,
                id="console-prompts-include-system",
                disabled=not has_system,
            )
            if not has_system:
                analysis_context.tooltip = SYSTEM_ANALYSIS_ABSENT_TOOLTIP
            widgets.extend(
                [
                    Vertical(
                        Static(
                            "",
                            id="console-prompts-recipe-save-confirmation",
                            markup=False,
                        ),
                        Button(
                            "Open Library",
                            id="console-prompts-open-saved-recipe",
                        ),
                        id="console-prompts-recipe-save-confirmation-panel",
                    ),
                    analysis_context,
                    Static(
                        SYSTEM_ANALYSIS_DISCLOSURE
                        if has_system
                        else SYSTEM_ANALYSIS_ABSENT_DISCLOSURE,
                        id="console-prompts-recipe-analysis-disclosure",
                        markup=False,
                    ),
                    fill,
                ]
            )
        widgets.extend(
            [
                Static("", id="console-prompts-improvement-status", markup=False),
                retry,
                PromptBlockEditor(
                    self._editor_state,
                    can_update_original=False,
                    show_back=False,
                    initially_hidden_block_ids=initially_hidden_block_ids,
                    id="console-prompts-editor",
                ),
            ]
        )
        await body.mount(
            VerticalScroll(
                *widgets,
                id="console-prompts-recipe-scroll",
            )
        )
        if artifact_type == "recipe":
            confirmation = self.query_one(
                "#console-prompts-recipe-save-confirmation-panel", Vertical
            )
            confirmation.display = False
            open_library = self.query_one(
                "#console-prompts-open-saved-recipe", Button
            )
            open_library.can_focus = False

    def _show_dirty_guard(self) -> None:
        guard = self.query_one("#console-prompts-dirty-guard", Vertical)
        if guard.display:
            return
        guard.add_class("visible")
        guard.display = True
        self.call_after_refresh(self._focus_widget, "console-prompts-keep-editing")

    def _hide_dirty_guard(self) -> None:
        guard = self.query_one("#console-prompts-dirty-guard", Vertical)
        guard.remove_class("visible")
        guard.display = False

    def _keep_editing(self) -> None:
        self._hide_dirty_guard()
        self.call_after_refresh(
            self._focus_widget,
            self.state.focus_for(self.state.mode),
        )

    def _improvement_is_cancelling(self) -> bool:
        worker = self._improvement_worker
        return bool(worker is not None and worker.is_cancelled and worker.is_running)

    def _request_close(self) -> None:
        if self._apply_in_progress:
            return
        self._recipe_selecting = False
        self._cancel_improvement_activation()
        if self._active_request_id is not None:
            self._cancel_improvement()
            return
        if self._improvement_is_cancelling():
            return
        if self.state.mode in {"edit", "recipe"} and self.state.dirty:
            self._show_dirty_guard()
            return
        self.dismiss_safe_once(None)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        guard = self.query_one("#console-prompts-dirty-guard", Vertical)
        if guard.display:
            if source == "escape":
                self._keep_editing()
            return
        self._request_close()

    @on(ConsolePromptsBrowse.ImproveRequested)
    async def _improve_requested(
        self, event: ConsolePromptsBrowse.ImproveRequested
    ) -> None:
        event.stop()
        await self.enter_mode(
            "improve",
            focus_id=self._improve_initial_focus_id(),
        )
        if self._improve_unavailable_reason:
            self._set_improvement_status(self._improve_unavailable_reason)

    @on(Checkbox.Changed, "#console-prompts-include-system")
    def _include_system_changed(self, event: Checkbox.Changed) -> None:
        """Remember the analysis permission after the path picker is replaced."""

        self._include_system_context = event.value

    @on(ConsolePromptsBrowse.ConfigureProviderRequested)
    async def _configure_provider_requested(
        self, event: ConsolePromptsBrowse.ConfigureProviderRequested
    ) -> None:
        event.stop()
        if self._configure_provider is not None:
            await _maybe_await(self._configure_provider())

    @on(ConsolePromptsBrowse.SourceChanged)
    async def _source_requested(
        self, event: ConsolePromptsBrowse.SourceChanged
    ) -> None:
        event.stop()
        await self.switch_source(event.source)

    @on(ConsolePromptsBrowse.QueryChanged)
    def _query_requested(self, event: ConsolePromptsBrowse.QueryChanged) -> None:
        event.stop()
        self.state = self.state.with_query(event.query).begin_search()
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
        self._debounce_timer = self.set_timer(
            0.2,
            lambda: self.run_worker(
                self.reload_browse(token_already_started=True),
                exclusive=False,
                group=f"console-prompts-search-{id(self)}",
            ),
        )

    @on(ConsolePromptsBrowse.PageRequested)
    async def _page_requested(self, event: ConsolePromptsBrowse.PageRequested) -> None:
        event.stop()
        self.state = self.state.with_page(event.page).begin_search()
        await self.reload_browse(token_already_started=True)

    @on(ConsolePromptsBrowse.RetryRequested)
    async def _retry_requested(
        self, event: ConsolePromptsBrowse.RetryRequested
    ) -> None:
        event.stop()
        await self.reload_browse()

    @on(ConsolePromptsBrowse.ArtifactSelected)
    async def _artifact_requested(
        self, event: ConsolePromptsBrowse.ArtifactSelected
    ) -> None:
        event.stop()
        await self.open_artifact(event.identifier)

    @on(PromptBlockEditor.BlockFieldChanged)
    def _block_changed(self, event: PromptBlockEditor.BlockFieldChanged) -> None:
        event.stop()
        self._editor_state = event.state
        self.mark_dirty()
        if self.state.mode != "recipe":
            self._sync_editor_host_gates()

    @on(PromptBlockEditor.BlockActionRequested)
    def _block_action(self, event: PromptBlockEditor.BlockActionRequested) -> None:
        event.stop()
        self._editor_state = event.state
        self.mark_dirty()
        if self.state.mode != "recipe":
            self._sync_editor_host_gates()

    @on(PromptBlockEditor.BackRequested)
    async def _editor_back(self, event: PromptBlockEditor.BackRequested) -> None:
        event.stop()
        await self._back_internal()

    @on(PromptBlockEditor.SaveAsPromptRequested)
    async def _save_prompt(
        self, event: PromptBlockEditor.SaveAsPromptRequested
    ) -> None:
        event.stop()
        await self._save_editor_state(event.state, artifact_type="prompt")

    @on(PromptBlockEditor.SaveAsRecipeRequested)
    async def _save_recipe(
        self, event: PromptBlockEditor.SaveAsRecipeRequested
    ) -> None:
        event.stop()
        await self._save_editor_state(event.state, artifact_type="recipe")

    @on(PromptBlockEditor.UpdateOriginalRequested)
    async def _update_original(
        self, event: PromptBlockEditor.UpdateOriginalRequested
    ) -> None:
        event.stop()
        if self._can_update_original():
            await self._save_editor_state(
                event.state,
                artifact_type="prompt",
                update_original=True,
            )

    @on(PromptBlockEditor.ApplyRequested)
    async def _apply_not_available(
        self, event: PromptBlockEditor.ApplyRequested
    ) -> None:
        event.stop()
        if self._manual_apply_resolution is None:
            await self._capture_manual_apply_target()
        if self.state.mode == "edit":
            decoded = self._decoded
            source = self.state.selected_source
            identity = self.state.selected_identity
            if (
                self.state.working_copy_unsaved
                or decoded is None
                or decoded.artifact_type != "prompt"
                or decoded.definition is None
                or source is None
                or identity is None
            ):
                return
            guard: ConsoleRecipeApplyGuard | ConsoleSavedPromptApplyGuard = (
                ConsoleSavedPromptApplyGuard(
                    source=source,
                    prompt_source_id=identity,
                    prompt_version=self.state.selected_version,
                    prompt_definition=decoded.definition,
                    prompt_fingerprint=_fingerprint_definition(decoded.definition),
                    record_usage=(
                        _fingerprint_definition(event.state.definition)
                        == _fingerprint_definition(decoded.definition)
                    ),
                    provider_resolution=self._manual_apply_resolution,
                )
            )
        elif self.state.mode == "recipe":
            definition = self._recipe_definition or replace(
                event.state.definition,
                kind="block_recipe",
            )
            guard = ConsoleRecipeApplyGuard(
                recipe_source=self._recipe_source,
                recipe_source_id=self._recipe_source_id,
                recipe_version=self._recipe_version,
                recipe_definition=definition,
                recipe_fingerprint=(
                    self._recipe_source_fingerprint
                    or _fingerprint_definition(definition)
                ),
                provider_resolution=self._manual_apply_resolution,
            )
        else:
            return
        generation = self._claim_apply_transaction()
        if generation is None:
            return
        result = self._result_for(
            user_text=event.user_prompt,
            system_text=event.system_prompt,
            apply_user=event.apply_user,
            apply_system=event.apply_system,
            captured=guard,
        )
        self.run_worker(
            self._coordinate_apply(result, guard, _generation=generation),
            exclusive=False,
            group=f"console-prompts-apply-{id(self)}",
        )

    def _cancel_improvement(self) -> None:
        if self._active_request_id is None:
            return
        self._set_improvement_status("Cancelling...")
        self._active_request_id = None
        self._sync_improve_gates()
        worker = self._improvement_worker
        if worker is not None:
            worker.cancel()

    async def _retry_persistence(self) -> None:
        result = self._pending_persistence_result
        if result is None or self._retry_improvement_persistence is None:
            return
        try:
            outcome = await _maybe_await(self._retry_improvement_persistence(result))
        except Exception:
            self._set_improvement_status(
                "Applied to this session, but could not save to the conversation."
            )
            return
        kind = str(getattr(outcome, "kind", ""))
        if kind == "applied":
            self._reset_persistence_retry()
            self.dismiss_safe_once(result)
            return
        if kind == "persistence_failed":
            self._set_improvement_status(
                "Applied to this session, but could not save to the conversation."
            )
            return
        self._reset_persistence_retry()
        self._set_improvement_status(
            str(
                getattr(outcome, "user_message", "")
                or "The live Console state changed. Review it before saving again."
            )
        )

    async def _save_editor_state(
        self,
        editor_state: PromptBlockEditorState,
        *,
        artifact_type: Literal["prompt", "recipe"],
        update_original: bool = False,
    ) -> None:
        artifact_label = artifact_type.title()
        if editor_state.issues:
            self.notify(
                f"{artifact_label} save unavailable — resolve the highlighted block errors first.",
                severity="warning",
            )
            return
        try:
            target_state = set_artifact_type(editor_state, artifact_type)
        except ValueError:
            message = (
                RECIPE_MAPPED_CONTEXT_BLOCKED_COPY
                if artifact_type == "recipe" and not editor_state.can_save_as_recipe
                else f"{artifact_label} save unavailable — the structured blocks are not valid for that artifact type."
            )
            self.notify(message, severity="warning")
            return
        if not self._supports_structured_save(artifact_type):
            source = self.state.selected_source or self.state.source
            if source not in self._capabilities_by_source:
                try:
                    self._capabilities_by_source[source] = await _maybe_await(
                        self._capabilities(source)
                    )
                except Exception:
                    pass
        if not self._supports_structured_save(artifact_type):
            self.notify(
                f"{artifact_label} save is unavailable — the selected source does not support this structured artifact kind.",
                severity="warning",
            )
            return
        record = self._selected_record or {}
        definition = target_state.definition
        payload = {
            "source": self.state.selected_source or self.state.source,
            "prompt_identifier": (
                self.state.selected_identity if update_original else None
            ),
            "expected_version": (
                self.state.selected_version if update_original else None
            ),
            "name": str(record.get("name") or "Untitled Prompt")
            + ("" if update_original else " copy"),
            "author": record.get("author"),
            "details": record.get("details"),
            "keywords": record.get("keywords"),
            "system_prompt": editor_state.compiled_system,
            "user_prompt": editor_state.compiled_user,
            "prompt_format": "structured",
            "prompt_schema_version": 2,
            "prompt_definition": _definition_mapping(definition),
            "artifact_type": artifact_type,
        }
        try:
            saved = await _maybe_await(self._save(**payload))
        except Exception:
            self.notify(
                "Prompt save failed — the working copy is unchanged. Retry after checking the selected source.",
                severity="error",
            )
            return
        if artifact_type == "recipe":
            self._show_recipe_saved_confirmation(saved)
            self.notify("Recipe saved as a new artifact.")
            return
        if not isinstance(saved, Mapping):
            self.notify(
                "Prompt saved, but its new identity was not returned. Reload the Library before updating it.",
                severity="warning",
            )
            return
        saved_record = dict(saved)
        try:
            identity = _saved_record_identifier(saved_record)
        except ValueError:
            self.notify(
                "Prompt saved, but its new identity was not returned. Reload the Library before updating it.",
                severity="warning",
            )
            return
        source_value = str(saved_record.get("backend") or "")
        selected_source: PromptSource = (
            source_value
            if source_value in {"local", "server"}
            else self.state.selected_source or self.state.source
        )  # type: ignore[assignment]
        self._selected_record = saved_record
        try:
            self._decoded = decode_prompt_artifact(saved_record)
        except (TypeError, ValueError):
            self._decoded = None
        if self._decoded is not None and not self._decoded.compatibility_stale:
            try:
                self.query_one(
                    "#console-prompts-compatibility-stale", Static
                ).display = False
            except NoMatches:
                pass
        self._editor_state = editor_state
        self.state = (
            self.state.select(
                identity=identity,
                version=_record_version(saved_record),
                source=selected_source,
                capabilities=self.state.selected_capabilities,
            )
            .as_unsaved_copy(False)
            .with_dirty(False)
        )
        self.query_one("#console-prompts-location", Static).update(
            self._location_copy(self.state.mode)
        )
        self._sync_editor_host_gates()
        self.notify("Prompt saved.")

    def _show_recipe_saved_confirmation(self, saved: Any) -> None:
        """Expose the Library destination when a saved Recipe has an identity."""

        try:
            panel = self.query_one(
                "#console-prompts-recipe-save-confirmation-panel", Vertical
            )
            confirmation = self.query_one(
                "#console-prompts-recipe-save-confirmation", Static
            )
            open_library = self.query_one(
                "#console-prompts-open-saved-recipe", Button
            )
        except NoMatches:
            return
        record = dict(saved) if isinstance(saved, Mapping) else {}
        name = str(record.get("name") or "Recipe").strip() or "Recipe"
        target: tuple[PromptSource, str] | None = None
        source_value = str(record.get("backend") or "")
        source: PromptSource = (
            source_value
            if source_value in {"local", "server"}
            else self.state.selected_source or self.state.source
        )  # type: ignore[assignment]
        local_id = record.get("local_id")
        if source == "local" and type(local_id) is int and local_id > 0:
            identity = str(local_id)
        elif source == "local" and str(record.get("id") or "").isdecimal():
            identity = str(record["id"])
        else:
            try:
                identity = _saved_record_identifier(record)
            except ValueError:
                identity = ""
        if source == "local" and identity.isdecimal():
            target = (source, identity)
        self._saved_recipe_library_target = target
        confirmation.update(f"{name} saved to Library > Prompts as a Recipe.")
        panel.display = True
        open_library.display = True
        open_library.disabled = target is None or self._open_library_prompt is None
        open_library.can_focus = not open_library.disabled
        if open_library.disabled:
            open_library.tooltip = (
                "Deep link available only for local recipes; open Library > "
                "Prompts and select its source."
                if identity
                else "Reload Library > Prompts to find this Recipe; its saved "
                "identity was not returned."
            )
            return
        open_library.tooltip = "Open the newly saved Recipe in Library > Prompts."
        self.call_after_refresh(self._focus_widget, "console-prompts-open-saved-recipe")

    @on(Button.Pressed)
    async def _shell_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "console-prompts-back":
            event.stop()
            await self._back_internal()
        elif button_id == "console-prompts-close":
            event.stop()
            self._request_close()
        elif button_id == "console-prompts-keep-editing":
            event.stop()
            self._keep_editing()
        elif button_id == "console-prompts-discard":
            event.stop()
            self._hide_dirty_guard()
            self.state = self.state.with_dirty(False)
            await self._back_internal(discard=True)
        elif button_id == "console-prompts-structured-recipe":
            event.stop()
            await self.enter_mode("recipe")
        elif button_id == "console-prompts-auto-improve":
            event.stop()
            self._begin_improvement_activation("auto")
        elif button_id == "console-prompts-review-improve":
            event.stop()
            self._begin_improvement_activation("review")
        elif button_id == "console-prompts-configure-provider":
            event.stop()
            if self._configure_provider is not None:
                await _maybe_await(self._configure_provider())
        elif button_id == "console-prompts-improvement-cancel":
            event.stop()
            self._cancel_improvement()
        elif button_id == "console-prompts-improvement-retry":
            event.stop()
            if self._last_improvement_mode is not None:
                self._begin_improvement(self._last_improvement_mode)
        elif button_id == "console-prompts-persistence-retry":
            event.stop()
            await self._retry_persistence()
        elif button_id == "console-prompts-open-saved-recipe":
            event.stop()
            target = self._saved_recipe_library_target
            if target is None or self._open_library_prompt is None:
                return
            opened = await _maybe_await(self._open_library_prompt(*target))
            if opened:
                self.dismiss_safe_once(None)
        elif button_id == "console-prompts-review-apply":
            event.stop()
            generation = self._claim_apply_transaction()
            if generation is not None:
                self.run_worker(
                    self._apply_review_candidate(generation),
                    exclusive=False,
                    group=f"console-prompts-apply-{id(self)}",
                )
        elif button_id == "console-prompts-recipe-outcome-first":
            event.stop()
            self._recipe_source = None
            self._recipe_source_id = "builtin:outcome-first"
            self._recipe_version = 0
            definition = outcome_first_recipe()
            self._recipe_source_fingerprint = _fingerprint_definition(definition)
            await self._mount_recipe_editor(
                definition,
                initially_hidden_block_ids=_OUTCOME_FIRST_OPTIONAL_BLOCK_IDS,
            )
        elif button_id == "console-prompts-recipe-blank":
            event.stop()
            self._recipe_source = None
            self._recipe_source_id = "builtin:blank"
            self._recipe_version = 0
            definition = blank_recipe()
            self._recipe_source_fingerprint = _fingerprint_definition(definition)
            await self._mount_recipe_editor(definition)
        elif button_id == "console-prompts-recipe-saved":
            event.stop()
            self._recipe_selecting = True
            await self.enter_mode("browse", focus_id="console-prompts-search")
        elif button_id == "console-prompts-recipe-fill":
            event.stop()
            self._begin_improvement_activation("recipe")
        elif button_id == "console-prompts-convert":
            event.stop()
            record = self._selected_record or {}
            decomposition = decompose_legacy_lanes(
                str(record.get("system_prompt") or ""),
                str(record.get("user_prompt") or ""),
            )
            self._editor_state = PromptBlockEditorState.from_definition(
                artifact_type="prompt",
                definition=decomposition.definition,
                system_origin=decomposition.system_origin,
                user_origin=decomposition.user_origin,
            )
            self.state = self.state.as_unsaved_copy(True).with_dirty(False)
            await self._mount_mode("edit")

    def action_back(self) -> None:
        if self.query_one("#console-prompts-dirty-guard", Vertical).display:
            self._keep_editing()
            return
        self.run_worker(
            self._back_internal(),
            exclusive=False,
            group=f"console-prompts-back-{id(self)}",
        )


__all__ = [
    "ConsolePromptsModal",
    "ConsolePromptsResult",
    "ConsolePromptsApplyOutcome",
    "ConsoleRecipeApplyGuard",
    "ConsoleSavedPromptApplyGuard",
]
