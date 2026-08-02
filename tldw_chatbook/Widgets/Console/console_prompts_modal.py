"""Unified Browse/Edit/Improve/Recipe shell for Console prompts."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Prompt_Management.prompt_artifact_codec import (
    decode_prompt_artifact,
    deserialize_definition,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    BlockArtifactDefinition,
    DecodedPromptArtifact,
)
from tldw_chatbook.Prompt_Management.prompt_legacy_decomposer import (
    decompose_legacy_lanes,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import PromptBlockEditor
from tldw_chatbook.Widgets.Prompts.prompt_block_editor_state import (
    PromptBlockEditorState,
)

from .console_prompts_browse import ConsolePromptsBrowse
from .console_prompts_state import (
    ConsolePromptsState,
    PromptBrowseResult,
    PromptModalMode,
    PromptSource,
)


_PER_PAGE = 10


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


def _record_identifier(record: Mapping[str, Any]) -> str:
    value = record.get("id", record.get("uuid", record.get("name", "")))
    if value in (None, ""):
        raise ValueError("Prompt detail has no source identity.")
    return str(value)


def _record_version(record: Mapping[str, Any]) -> int | None:
    value = record.get("version", record.get("optimistic_version"))
    return value if type(value) is int else None


def _saved_record_identifier(record: Mapping[str, Any]) -> str:
    """Return a durable identity from a normalized save response."""
    value = record.get("id") or record.get("uuid") or record.get("source_id")
    if value in (None, ""):
        raise ValueError("Saved Prompt response has no durable identity.")
    return str(value)


class ConsolePromptsModal(ModalScreen[None]):
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
    #console-prompts-footer { width: 100%; height: 3; align: right middle; }
    #console-prompts-dirty-guard {
        display: none; width: 100%; height: auto; padding: 1;
        border: round $warning; background: $warning 10%;
    }
    #console-prompts-dirty-guard.visible { display: block; }
    """

    BINDINGS = [("escape", "back", "Back")]

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
        self.state = ConsolePromptsState.initial()
        self.browse_result = PromptBrowseResult(
            source="local", items=(), page=1, total_pages=1, total_items=0
        )
        self._capabilities_by_source: dict[str, object] = {}
        self._debounce_timer: Timer | None = None
        self._editor_state: PromptBlockEditorState | None = None
        self._selected_record: Mapping[str, Any] | None = None
        self._decoded: DecodedPromptArtifact | None = None
        self._compatibility_state = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="console-prompts-modal"):
            with Vertical(id="console-prompts-header"):
                yield Static(
                    "Prompt Workbench", id="console-prompts-title", markup=False
                )
                yield Static(
                    "Prompts / Browse", id="console-prompts-location", markup=False
                )
            with Vertical(id="console-prompts-body"):
                yield self._browse_widget()
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
            can_configure_provider=self._configure_provider is not None,
            id="console-prompts-browse",
        )

    def on_mount(self) -> None:
        self._set_responsive(self.app.size.width, self.app.size.height)
        self.run_worker(
            self.reload_browse(),
            exclusive=False,
            group=f"console-prompts-load-{id(self)}",
        )

    def on_resize(self, event: Any) -> None:
        self._set_responsive(event.size.width, event.size.height)

    def on_unmount(self) -> None:
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
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

    def _focus_widget(self, widget_id: str | None) -> None:
        if not widget_id:
            return
        try:
            self.query_one(f"#{widget_id}").focus()
        except NoMatches:
            return

    async def enter_mode(
        self, mode: PromptModalMode, *, focus_id: str | None = None
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported prompt modal mode: {mode}")
        self._remember_current_focus()
        self.state = self.state.enter_mode(mode)
        await self._mount_mode(mode)
        self.call_after_refresh(
            self._focus_widget,
            focus_id or self.state.focus_for(mode),
        )

    async def _back_internal(self, *, discard: bool = False) -> None:
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
                editor = PromptBlockEditor(
                    self._editor_state,
                    can_update_original=self._can_update_original(),
                    id="console-prompts-editor",
                )
                await body.mount(editor)
                self._sync_editor_host_gates(editor)
            else:
                await self._mount_compatibility(body)
        elif mode == "improve":
            await body.mount(self._improve_placeholder())
        else:
            await body.mount(self._recipe_placeholder())

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

    def _improve_placeholder(self) -> Vertical:
        container = Vertical(id="console-prompts-improve-mode")
        container.compose_add_child(
            Static(
                "Choose how the current unsent message should be improved. No model call starts until you choose a path.",
                markup=False,
            )
        )
        for label, button_id in (
            ("Analyze and auto-improve", "console-prompts-auto-improve"),
            ("Analyze and review", "console-prompts-review-improve"),
            (
                "Create or follow a structured recipe",
                "console-prompts-structured-recipe",
            ),
        ):
            button = Button(
                label,
                id=button_id,
                disabled=bool(self._improve_unavailable_reason),
            )
            if self._improve_unavailable_reason:
                button.tooltip = self._improve_unavailable_reason
            container.compose_add_child(button)
        return container

    def _recipe_placeholder(self) -> Vertical:
        return Vertical(
            Static(
                "Structured recipe authoring opens here. Browse state remains intact while this working mode is active.",
                markup=False,
            ),
            id="console-prompts-recipe-mode",
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
        capabilities = self.state.selected_capabilities
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
        editor.set_update_original_available(self._can_update_original())
        has_issues = bool(editor.state.issues)
        for artifact_type, selector, kind in (
            ("prompt", "#prompt-editor-save-prompt", "block_prompt"),
            ("recipe", "#prompt-editor-save-recipe", "block_recipe"),
        ):
            button = editor.query_one(selector, Button)
            supported = self._supports_structured_save(artifact_type)
            button.disabled = has_issues or not supported
            if not supported:
                button.tooltip = (
                    f"Save unavailable — {self.state.selected_source or self.state.source} "
                    f"does not support {kind}; switch source or keep editing."
                )
        update = editor.query_one("#prompt-editor-update-original", Button)
        update.disabled = has_issues or not self._can_update_original()
        if not self._can_update_original():
            update.tooltip = (
                "Update unavailable — this source does not support conditional updates "
                "for this version; save as new."
            )
        else:
            update.tooltip = None
        apply_button = editor.query_one("#prompt-editor-apply", Button)
        apply_button.disabled = True
        apply_button.tooltip = "Applying to the composer is unavailable in this stage; save the Prompt instead."
        apply_reason = editor.query_one("#prompt-editor-apply-reason", Static)
        apply_reason.update(
            "Apply unavailable in this stage — save the Prompt instead."
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

    def mark_dirty(self) -> None:
        self.state = self.state.with_dirty(True)

    def _show_dirty_guard(self) -> None:
        guard = self.query_one("#console-prompts-dirty-guard", Vertical)
        guard.add_class("visible")
        guard.display = True
        self.call_after_refresh(self._focus_widget, "console-prompts-keep-editing")

    def _hide_dirty_guard(self) -> None:
        guard = self.query_one("#console-prompts-dirty-guard", Vertical)
        guard.remove_class("visible")
        guard.display = False

    @on(ConsolePromptsBrowse.ImproveRequested)
    async def _improve_requested(
        self, event: ConsolePromptsBrowse.ImproveRequested
    ) -> None:
        event.stop()
        if not self._improve_unavailable_reason:
            await self.enter_mode("improve")

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
        self._sync_editor_host_gates()

    @on(PromptBlockEditor.BlockActionRequested)
    def _block_action(self, event: PromptBlockEditor.BlockActionRequested) -> None:
        event.stop()
        self._editor_state = event.state
        self.mark_dirty()
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
    def _apply_not_available(self, event: PromptBlockEditor.ApplyRequested) -> None:
        event.stop()

    async def _save_editor_state(
        self,
        editor_state: PromptBlockEditorState,
        *,
        artifact_type: str,
        update_original: bool = False,
    ) -> None:
        if not self._supports_structured_save(artifact_type):
            self.notify(
                "Prompt save is unavailable — the selected source does not support this structured artifact kind.",
                severity="warning",
            )
            return
        record = self._selected_record or {}
        kind = "block_recipe" if artifact_type == "recipe" else "block_prompt"
        definition = replace(editor_state.definition, kind=kind)
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

    @on(Button.Pressed)
    async def _shell_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "console-prompts-back":
            event.stop()
            await self._back_internal()
        elif button_id == "console-prompts-close":
            event.stop()
            if self.state.mode in {"edit", "recipe"} and self.state.dirty:
                self._show_dirty_guard()
            else:
                self.dismiss(None)
        elif button_id == "console-prompts-keep-editing":
            event.stop()
            self._hide_dirty_guard()
            self.call_after_refresh(
                self._focus_widget,
                self.state.focus_for(self.state.mode),
            )
        elif button_id == "console-prompts-discard":
            event.stop()
            self._hide_dirty_guard()
            self.state = self.state.with_dirty(False)
            await self._back_internal(discard=True)
        elif button_id == "console-prompts-structured-recipe":
            event.stop()
            await self.enter_mode("recipe")
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
            self._hide_dirty_guard()
            return
        self.run_worker(
            self._back_internal(),
            exclusive=False,
            group=f"console-prompts-back-{id(self)}",
        )


__all__ = ["ConsolePromptsModal"]
