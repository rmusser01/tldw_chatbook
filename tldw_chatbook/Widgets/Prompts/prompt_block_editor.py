"""Incremental Textual editor for schema-v2 Prompt and Recipe blocks."""

# THESIS: Edit structured prompts without treating mounted text editors as disposable.
# OWN-WORLD: Neon Workbench tokens, flat lanes, semantic state labels, stable focus.
# STORY: Scan System then User, resolve errors in place, choose lanes, save or apply.
# FIRST VIEWPORT: Stacked expandable lanes fill the work area; action truth stays pinned.
# FORM: Dense Operate-mode extension of the incumbent Console; no new-world seed.

from __future__ import annotations

import re
from typing import Literal

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Select,
    Static,
    TextArea,
)

from tldw_chatbook.Prompt_Management.prompt_artifact_models import PromptBlock
from tldw_chatbook.Widgets.Prompts.prompt_block_editor_state import (
    LaneId,
    PromptBlockEditorState,
    PromptBlockValidationIssue,
    add_block,
    can_duplicate_block,
    delete_block,
    duplicate_block,
    move_block,
    update_block,
)


BlockField = Literal["title", "syntax", "xml_tag", "content"]
BlockAction = Literal["move_up", "move_down", "duplicate", "delete", "add"]

_NARROW_WIDTH = 90
_STACKED_FOOTER_WIDTH = 120
_SAFE_WIDGET_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
RECIPE_MAPPED_CONTEXT_BLOCKED_COPY = (
    "Recipe save unavailable — delete the mapped Additional context block first."
)


class PromptBlockCard(Vertical):
    """One stable mounted block whose controls are patched in place."""

    def __init__(
        self,
        block: PromptBlock,
        *,
        token: str,
        issues: tuple[PromptBlockValidationIssue, ...],
        dirty: bool,
        is_first: bool,
        is_last: bool,
    ) -> None:
        super().__init__(
            id=f"prompt-block-{token}",
            classes="prompt-block-card",
        )
        self.block_id = block.id
        self.token = token
        self._block = block
        self._issues = issues
        self._dirty = dirty
        self._is_first = is_first
        self._is_last = is_last

    def compose(self) -> ComposeResult:
        block = self._block
        with Horizontal(classes="prompt-block-header"):
            with Horizontal(classes="prompt-block-metadata"):
                yield Input(
                    block.title,
                    placeholder="Block title",
                    id=f"prompt-block-title-{self.token}",
                    classes="prompt-block-title",
                )
                yield Select(
                    (
                        ("Free-form", "freeform"),
                        ("Markdown", "markdown"),
                        ("XML", "xml"),
                    ),
                    value=block.syntax,
                    allow_blank=False,
                    id=f"prompt-block-syntax-{self.token}",
                    classes="prompt-block-syntax",
                )
            with Horizontal(classes="prompt-block-actions"):
                yield Button(
                    "↑ Up",
                    id=f"prompt-block-move-up-{self.token}",
                    classes="prompt-block-action",
                    disabled=self._is_first,
                )
                yield Button(
                    "↓ Down",
                    id=f"prompt-block-move-down-{self.token}",
                    classes="prompt-block-action",
                    disabled=self._is_last,
                )
                duplicate = Button(
                    "Duplicate",
                    id=f"prompt-block-duplicate-{self.token}",
                    classes="prompt-block-action",
                    disabled=not can_duplicate_block(block.id),
                )
                if duplicate.disabled:
                    duplicate.tooltip = (
                        "Duplicate unavailable — mapped Additional context uses a "
                        "reserved identity."
                    )
                yield duplicate
                yield Button(
                    "Delete",
                    id=f"prompt-block-delete-{self.token}",
                    classes="prompt-block-action",
                )
        xml_tag = Input(
            block.xml_tag or "",
            placeholder="XML wrapper tag",
            id=f"prompt-block-xml-tag-{self.token}",
            classes="prompt-block-xml-tag",
        )
        xml_tag.display = block.syntax == "xml"
        yield xml_tag
        yield TextArea(
            block.content,
            soft_wrap=True,
            show_line_numbers=False,
            placeholder="Prompt content",
            id=f"prompt-block-content-{self.token}",
            classes="prompt-block-content",
        )
        yield Static(
            self._issue_copy(),
            id=f"prompt-block-issue-{self.token}",
            classes="prompt-block-issue",
            markup=False,
        )
        yield Static(
            self._status_copy(),
            id=f"prompt-block-status-{self.token}",
            classes="prompt-block-status",
            markup=False,
        )

    def sync(
        self,
        block: PromptBlock,
        *,
        issues: tuple[PromptBlockValidationIssue, ...],
        dirty: bool,
        is_first: bool,
        is_last: bool,
    ) -> None:
        """Patch changed controls without replacing this card or its TextArea."""
        self._block = block
        self._issues = issues
        self._dirty = dirty
        self._is_first = is_first
        self._is_last = is_last

        title = self.query_one(f"#prompt-block-title-{self.token}", Input)
        if title.value != block.title:
            with title.prevent(Input.Changed):
                title.value = block.title

        syntax = self.query_one(f"#prompt-block-syntax-{self.token}", Select)
        if syntax.value != block.syntax:
            with syntax.prevent(Select.Changed):
                syntax.value = block.syntax

        xml_tag = self.query_one(f"#prompt-block-xml-tag-{self.token}", Input)
        next_xml_tag = block.xml_tag or ""
        if xml_tag.value != next_xml_tag:
            with xml_tag.prevent(Input.Changed):
                xml_tag.value = next_xml_tag
        xml_tag.display = block.syntax == "xml"

        content = self.query_one(f"#prompt-block-content-{self.token}", TextArea)
        if content.text != block.content:
            with content.prevent(TextArea.Changed):
                content.load_text(block.content)

        self.query_one(f"#prompt-block-issue-{self.token}", Static).update(
            self._issue_copy()
        )
        self.query_one(f"#prompt-block-status-{self.token}", Static).update(
            self._status_copy()
        )
        self.query_one(
            f"#prompt-block-move-up-{self.token}", Button
        ).disabled = is_first
        self.query_one(
            f"#prompt-block-move-down-{self.token}", Button
        ).disabled = is_last
        duplicate = self.query_one(f"#prompt-block-duplicate-{self.token}", Button)
        duplicate.disabled = not can_duplicate_block(block.id)
        duplicate.tooltip = (
            "Duplicate unavailable — mapped Additional context uses a reserved identity."
            if duplicate.disabled
            else None
        )
        self.set_class(bool(issues), "invalid")
        self.set_class(dirty, "dirty")

    def _issue_copy(self) -> str:
        if not self._issues:
            return ""
        return "Invalid — " + " ".join(issue.message for issue in self._issues)

    def _status_copy(self) -> str:
        parts = ["Unsaved changes" if self._dirty else "Valid"]
        if self._issues:
            parts[0] = "Invalid"
        if self._is_first:
            parts.append("first in lane; Move up unavailable")
        if self._is_last:
            parts.append("last in lane; Move down unavailable")
        return " · ".join(parts)


class PromptBlockEditor(Vertical):
    """Edit two canonical lanes while preserving unaffected native editor state."""

    DEFAULT_CSS = """
    /* INTENTIONAL widget-local palette (TASK-16811 audit): these are this
       editor's own design values, not parse fallbacks. They deliberately
       shadow the app tokens for this source ($ds-focus-bg: $accent 12%,
       $ds-action-focus: $accent), and $ds-surface-field exists only here. */
    $ds-surface-panel: $panel;
    $ds-surface-raised: $surface;
    $ds-surface-field: $surface-darken-1;
    $ds-grid-line: $surface-lighten-1;
    $ds-text-primary: $text;
    $ds-text-muted: $text-muted;
    $ds-text-disabled: $text-disabled;
    $ds-action-focus: $accent;
    $ds-focus-bg: $accent 12%;
    $ds-status-error: $error;

    PromptBlockEditor {
        width: 100%; height: 100%; min-height: 0;
        background: $ds-surface-panel; color: $ds-text-primary;
    }
    PromptBlockEditor.embedded {
        height: auto;
        min-height: 0;
    }
    #prompt-editor-scroll {
        width: 100%; height: 1fr; min-height: 0;
        overflow-y: auto; overflow-x: hidden; scrollbar-size: 1 1;
        scrollbar-background: $ds-surface-panel;
        scrollbar-color: $ds-grid-line;
        scrollbar-color-active: $ds-action-focus;
    }
    #prompt-editor-body {
        width: 100%; height: auto; min-height: 0;
    }
    .prompt-lane {
        width: 100%; height: auto; margin: 0 0 1 0;
        border: round $ds-grid-line; background: $ds-surface-panel;
    }
    .prompt-lane:focus-within { border: round $ds-action-focus; }
    .prompt-lane-blocks { width: 100%; height: auto; min-height: 1; }
    .prompt-lane-empty {
        width: 100%; height: auto; padding: 0 1; color: $ds-text-muted;
    }
    .prompt-block-card {
        width: 100%; height: auto; padding: 0 1 1 1;
        border-bottom: solid $ds-grid-line; background: $ds-surface-panel;
    }
    .prompt-block-card.invalid { background: $ds-status-error 8%; }
    .prompt-block-header { width: 100%; height: auto; min-height: 3; }
    .prompt-block-metadata { width: 1fr; min-width: 20; height: 3; }
    .prompt-block-actions { width: auto; height: 3; }

    .prompt-block-title,
    .prompt-block-syntax,
    .prompt-block-xml-tag {
        height: 3; border: none; border-left: solid $ds-grid-line;
        background: $ds-surface-field;
    }
    .prompt-block-title { width: 1fr; min-width: 12; }
    .prompt-block-syntax { width: 16; }
    .prompt-block-xml-tag { width: 100%; }
    .prompt-block-content {
        width: 100%; height: 6; min-height: 4;
        border: round $ds-grid-line; background: $ds-surface-field;
    }
    .prompt-block-title:focus,
    .prompt-block-syntax:focus,
    .prompt-block-xml-tag:focus,
    .prompt-block-content:focus {
        outline: heavy $ds-action-focus; background: $ds-focus-bg;
    }

    .prompt-lane-add,
    .prompt-block-action,
    .prompt-editor-action {
        height: 3; border: none;
        background: $ds-surface-raised; color: $ds-text-primary;
    }
    .prompt-lane-add { width: 100%; }
    .prompt-block-action { width: auto; min-width: 8; padding: 0 1; }
    .prompt-editor-action { width: auto; min-width: 6; padding: 0; }
    .prompt-lane-add:focus,
    .prompt-block-action:focus,
    .prompt-editor-action:focus { outline: heavy $ds-action-focus; }
    .prompt-block-action:disabled,
    .prompt-editor-action:disabled {
        background: $ds-surface-field; color: $ds-text-disabled;
    }

    .prompt-block-issue,
    .prompt-block-status,
    #prompt-editor-validation,
    #prompt-editor-apply-reason,
    #prompt-editor-update-reason {
        width: 100%; height: auto; min-height: 1; color: $ds-text-muted;
    }
    .prompt-block-issue,
    #prompt-editor-validation.invalid,
    #prompt-editor-apply-reason.blocked { color: $ds-status-error; }

    #prompt-editor-footer {
        width: 100%; height: auto; min-height: 3; layout: horizontal;
        border-top: solid $ds-grid-line; background: $ds-surface-panel;
    }
    #prompt-editor-footer.two-row { layout: vertical; min-height: 6; }
    .prompt-editor-footer-row { width: auto; height: 3; min-height: 3; }
    #prompt-editor-footer.two-row .prompt-editor-footer-row { width: 100%; }
    #prompt-editor-lane-options { align: left middle; }
    #prompt-editor-actions { width: 1fr; align: right middle; }
    #prompt-editor-apply {
        background: $ds-action-focus 35%; text-style: bold;
    }

    PromptBlockEditor.-narrow .prompt-block-header {
        layout: vertical; min-height: 6;
    }
    PromptBlockEditor.-narrow .prompt-block-metadata,
    PromptBlockEditor.-narrow .prompt-block-actions { width: 100%; }
    PromptBlockEditor.-narrow .prompt-block-action {
        width: 1fr; min-width: 0; padding: 0;
    }
    """

    BINDINGS = [
        ("ctrl+enter", "apply", "Apply selected lanes"),
        ("ctrl+s", "save_prompt", "Save as Prompt"),
    ]

    class BlockFieldChanged(Message):
        """A user edited one stable block field."""

        def __init__(
            self,
            block_id: str,
            field: BlockField,
            value: str,
            state: PromptBlockEditorState,
        ) -> None:
            self.block_id = block_id
            self.field = field
            self.value = value
            self.state = state
            super().__init__()

    class BlockActionRequested(Message):
        """A structural block action completed in the working copy."""

        def __init__(
            self,
            action: BlockAction,
            *,
            block_id: str | None,
            lane_id: LaneId,
            state: PromptBlockEditorState,
        ) -> None:
            self.action = action
            self.block_id = block_id
            self.lane_id = lane_id
            self.state = state
            super().__init__()

    class BackRequested(Message):
        """The hosting surface should navigate back."""

    class SaveAsPromptRequested(Message):
        """The valid working copy should be saved as a Prompt."""

        def __init__(self, state: PromptBlockEditorState) -> None:
            self.state = state
            super().__init__()

    class SaveAsRecipeRequested(Message):
        """The valid working copy should be saved as a Recipe."""

        def __init__(self, state: PromptBlockEditorState) -> None:
            self.state = state
            super().__init__()

    class UpdateOriginalRequested(Message):
        """The valid working copy should replace its guarded source version."""

        def __init__(self, state: PromptBlockEditorState) -> None:
            self.state = state
            super().__init__()

    class ApplyRequested(Message):
        """Apply only the explicitly selected non-empty lanes."""

        def __init__(
            self,
            *,
            apply_system: bool,
            apply_user: bool,
            system_prompt: str | None,
            user_prompt: str | None,
            state: PromptBlockEditorState,
        ) -> None:
            self.apply_system = apply_system
            self.apply_user = apply_user
            self.system_prompt = system_prompt
            self.user_prompt = user_prompt
            self.state = state
            super().__init__()

    def __init__(
        self,
        state: PromptBlockEditorState,
        *,
        can_update_original: bool = False,
        allow_apply_system: bool = True,
        apply_system_unavailable_reason: str = "",
        embedded: bool = False,
        host_owned_lifecycle: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._can_update_original = can_update_original
        self._allow_apply_system = bool(allow_apply_system)
        self._apply_system_unavailable_reason = apply_system_unavailable_reason.strip()
        self._embedded = bool(embedded)
        self._host_owned_lifecycle = bool(host_owned_lifecycle)
        self.set_class(self._embedded, "embedded")
        self._block_widgets: dict[str, PromptBlockCard] = {}

    @property
    def state(self) -> PromptBlockEditorState:
        """Return the immutable editor working copy."""
        return self._state

    def set_update_original_available(self, available: bool) -> None:
        """Refresh a host-owned guarded-update capability in place."""
        available = bool(available)
        if available == self._can_update_original:
            return
        self._can_update_original = available
        if self.is_mounted:
            self._sync_footer()

    @staticmethod
    def widget_token_for_block_id(block_id: str) -> str:
        """Return a deterministic legal token without title/index dependence."""
        if _SAFE_WIDGET_TOKEN.fullmatch(block_id) and not block_id.startswith(
            "encoded-"
        ):
            return block_id
        return f"encoded-{block_id.encode('utf-8').hex()}"

    def compose(self) -> ComposeResult:
        """Compose the block editor with the layout required by its host.

        Embedded editors yield a natural-height body so the parent owns vertical
        scrolling. Standalone editors yield their own vertical scroll container.

        Yields:
            Widgets for each editable Prompt or Recipe lane and block.
        """
        body_class = Vertical if self._embedded else VerticalScroll
        body_id = "prompt-editor-body" if self._embedded else "prompt-editor-scroll"
        with body_class(id=body_id):
            for lane in self._state.definition.lanes:
                with Collapsible(
                    title=f"{lane.id.title()} · {len(lane.blocks)} blocks",
                    collapsed=not bool(lane.blocks),
                    id=f"prompt-lane-{lane.id}",
                    classes="prompt-lane",
                ):
                    with Vertical(
                        id=f"prompt-lane-{lane.id}-blocks",
                        classes="prompt-lane-blocks",
                    ):
                        for index, block in enumerate(lane.blocks):
                            yield self._new_block_card(
                                block,
                                is_first=index == 0,
                                is_last=index == len(lane.blocks) - 1,
                            )
                        empty = Static(
                            "Empty — add a block to build this lane.",
                            id=f"prompt-lane-{lane.id}-empty",
                            classes="prompt-lane-empty",
                            markup=False,
                        )
                        empty.display = not lane.blocks
                        yield empty
                        yield Button(
                            f"Add {lane.id.title()} block",
                            id=f"prompt-lane-add-{lane.id}",
                            classes="prompt-lane-add",
                        )

        with Vertical(id="prompt-editor-status"):
            yield Static("", id="prompt-editor-validation", markup=False)
            yield Static("", id="prompt-editor-apply-reason", markup=False)
            yield Static("", id="prompt-editor-update-reason", markup=False)

        with Horizontal(id="prompt-editor-footer"):
            lane_options = Horizontal(
                id="prompt-editor-lane-options",
                classes="prompt-editor-footer-row",
            )
            lane_options.display = not self._host_owned_lifecycle
            with lane_options:
                system = Checkbox(
                    "Apply system prompt to this session",
                    value=False,
                    id="prompt-editor-apply-system",
                    disabled=not self._allow_apply_system,
                )
                if not self._allow_apply_system:
                    system.tooltip = self._apply_system_unavailable_reason
                system.display = bool(self._state.compiled_system)
                yield system
                yield Checkbox(
                    "Apply User",
                    value=bool(self._state.compiled_user),
                    id="prompt-editor-apply-user",
                )
            with Horizontal(
                id="prompt-editor-actions",
                classes="prompt-editor-footer-row",
            ):
                back = Button(
                    "Back",
                    id="prompt-editor-back",
                    classes="prompt-editor-action",
                )
                back.display = not self._host_owned_lifecycle
                yield back
                yield Button(
                    "Save Prompt",
                    id="prompt-editor-save-prompt",
                    classes="prompt-editor-action",
                )
                yield Button(
                    "Save Recipe",
                    id="prompt-editor-save-recipe",
                    classes="prompt-editor-action",
                )
                update_original = Button(
                    "Update original",
                    id="prompt-editor-update-original",
                    classes="prompt-editor-action",
                    disabled=not self._can_update_original,
                )
                update_original.display = not self._host_owned_lifecycle
                yield update_original
                apply = Button(
                    "Apply",
                    id="prompt-editor-apply",
                    classes="prompt-editor-action",
                )
                apply.display = not self._host_owned_lifecycle
                yield apply

    def on_mount(self) -> None:
        self._set_responsive(self.size.width or self.app.size.width)
        try:
            self._sync_footer()
        except NoMatches:
            self.call_after_refresh(self._sync_footer)

    def on_resize(self, event: events.Resize) -> None:
        self._set_responsive(event.size.width)

    def _set_responsive(self, width: int) -> None:
        narrow = width < _NARROW_WIDTH
        self.set_class(narrow, "-narrow")
        try:
            self.query_one("#prompt-editor-footer").set_class(
                width < _STACKED_FOOTER_WIDTH,
                "two-row",
            )
        except NoMatches:
            return

    def _new_block_card(
        self, block: PromptBlock, *, is_first: bool, is_last: bool
    ) -> PromptBlockCard:
        token = self.widget_token_for_block_id(block.id)
        card = PromptBlockCard(
            block,
            token=token,
            issues=self._issues_for(block.id),
            dirty=block.id in self._state.dirty_block_ids,
            is_first=is_first,
            is_last=is_last,
        )
        self._block_widgets[block.id] = card
        return card

    async def replace_block_state(
        self, block_id: str, state: PromptBlockEditorState
    ) -> None:
        """Incrementally reconcile state without rebuilding unaffected cards."""
        old_definition = self._state.definition
        old_shapes = tuple(
            (lane.id, tuple(block.id for block in lane.blocks))
            for lane in old_definition.lanes
        )
        new_shapes = tuple(
            (lane.id, tuple(block.id for block in lane.blocks))
            for lane in state.definition.lanes
        )
        old_system_nonempty = bool(self._state.compiled_system)
        old_user_nonempty = bool(self._state.compiled_user)
        old_lane_counts = {lane.id: len(lane.blocks) for lane in old_definition.lanes}
        self._state = state

        if old_shapes != new_shapes:
            for lane in state.definition.lanes:
                await self._reconcile_lane(
                    lane.id,
                    previous_count=old_lane_counts[lane.id],
                )
        else:
            located = self._find_block(block_id)
            if located is not None:
                lane_id, index, block = located
                self._sync_card(
                    block,
                    is_first=index == 0,
                    is_last=index
                    == len(
                        self._state.definition.lanes[
                            0 if lane_id == "system" else 1
                        ].blocks
                    )
                    - 1,
                )

        self._sync_lane_defaults(
            old_system_nonempty=old_system_nonempty,
            old_user_nonempty=old_user_nonempty,
        )
        self._sync_footer()

    async def _reconcile_lane(self, lane_id: LaneId, *, previous_count: int) -> None:
        lane = self._state.definition.lanes[0 if lane_id == "system" else 1]
        desired_ids = [block.id for block in lane.blocks]
        desired_set = set(desired_ids)
        body = self.query_one(f"#prompt-lane-{lane_id}-blocks", Vertical)

        for stale_id in [
            block_id
            for block_id, card in self._block_widgets.items()
            if card.parent is body and block_id not in desired_set
        ]:
            card = self._block_widgets.pop(stale_id)
            await card.remove()

        for index, block in enumerate(lane.blocks):
            card = self._block_widgets.get(block.id)
            if card is None:
                card = self._new_block_card(
                    block,
                    is_first=index == 0,
                    is_last=index == len(lane.blocks) - 1,
                )
                await body.mount(card, before=f"#prompt-lane-{lane_id}-empty")

        previous: Widget | None = None
        for index, block in enumerate(lane.blocks):
            card = self._block_widgets[block.id]
            if previous is None:
                body.move_child(card, before=0)
            else:
                body.move_child(card, after=previous)
            self._sync_card(
                block,
                is_first=index == 0,
                is_last=index == len(lane.blocks) - 1,
            )
            previous = card

        self.query_one(
            f"#prompt-lane-{lane_id}-empty", Static
        ).display = not lane.blocks
        collapsible = self.query_one(f"#prompt-lane-{lane_id}", Collapsible)
        collapsible.title = f"{lane.id.title()} · {len(lane.blocks)} blocks"
        if previous_count == 0 and lane.blocks:
            collapsible.collapsed = False

    def _sync_card(self, block: PromptBlock, *, is_first: bool, is_last: bool) -> None:
        self._block_widgets[block.id].sync(
            block,
            issues=self._issues_for(block.id),
            dirty=block.id in self._state.dirty_block_ids,
            is_first=is_first,
            is_last=is_last,
        )

    def _sync_lane_defaults(
        self, *, old_system_nonempty: bool, old_user_nonempty: bool
    ) -> None:
        system_nonempty = bool(self._state.compiled_system)
        user_nonempty = bool(self._state.compiled_user)
        system = self.query_one("#prompt-editor-apply-system", Checkbox)
        user = self.query_one("#prompt-editor-apply-user", Checkbox)
        system.display = system_nonempty
        if not system_nonempty:
            with system.prevent(Checkbox.Changed):
                system.value = False
        elif not old_system_nonempty:
            with system.prevent(Checkbox.Changed):
                system.value = False
        if not user_nonempty:
            with user.prevent(Checkbox.Changed):
                user.value = False
        elif not old_user_nonempty:
            with user.prevent(Checkbox.Changed):
                user.value = True

    def _sync_footer(self) -> None:
        if not self.is_attached:
            return
        issue_count = len(self._state.issues)
        validation = self.query_one("#prompt-editor-validation", Static)
        if issue_count:
            validation.update(
                f"Invalid · {issue_count} blocking error"
                f"{'s' if issue_count != 1 else ''} — correct the highlighted block."
            )
        else:
            block_count = sum(len(lane.blocks) for lane in self._state.definition.lanes)
            validation.update(f"Valid · {block_count} blocks")
        validation.set_class(bool(issue_count), "invalid")

        system = self.query_one("#prompt-editor-apply-system", Checkbox)
        user = self.query_one("#prompt-editor-apply-user", Checkbox)
        selected_system = bool(
            self._allow_apply_system and system.value and self._state.compiled_system
        )
        selected_user = bool(user.value and self._state.compiled_user)
        no_selected_content = not (selected_system or selected_user)

        apply_reason = self.query_one("#prompt-editor-apply-reason", Static)
        if issue_count:
            reason = "Apply unavailable — resolve the block errors above."
        elif not self._state.compiled_system and not self._state.compiled_user:
            reason = "Apply unavailable — add content to a System or User block."
        elif no_selected_content:
            reason = (
                self._apply_system_unavailable_reason
                if not self._allow_apply_system
                and self._state.compiled_system
                and not self._state.compiled_user
                else "Apply unavailable — select a non-empty lane."
            )
        else:
            reason = (
                f"{self._apply_system_unavailable_reason} "
                "Ready — Apply inserts the selected User lane."
                if not self._allow_apply_system
                and self._state.compiled_system
                and self._apply_system_unavailable_reason
                else "Ready — Apply changes only the selected non-empty lanes."
            )
        apply_reason.update(reason)
        apply_reason.set_class(issue_count > 0 or no_selected_content, "blocked")

        self.query_one("#prompt-editor-save-prompt", Button).disabled = bool(
            issue_count
        )
        save_recipe = self.query_one("#prompt-editor-save-recipe", Button)
        recipe_conversion_blocked = not self._state.can_save_as_recipe
        save_recipe.disabled = bool(issue_count) or recipe_conversion_blocked
        save_recipe.tooltip = (
            RECIPE_MAPPED_CONTEXT_BLOCKED_COPY if recipe_conversion_blocked else None
        )
        update = self.query_one("#prompt-editor-update-original", Button)
        update.disabled = bool(issue_count) or not self._can_update_original
        self.query_one("#prompt-editor-apply", Button).disabled = bool(
            issue_count or no_selected_content
        )
        update_reason = self.query_one("#prompt-editor-update-reason", Static)
        if not self._can_update_original:
            update_copy = (
                "Update unavailable — this source has no guarded version update; "
                "save as new."
            )
        elif issue_count:
            update_copy = "Update unavailable — resolve the block errors above."
        else:
            update_copy = ""
        update_reason.update(update_copy)

    def _issues_for(self, block_id: str) -> tuple[PromptBlockValidationIssue, ...]:
        return tuple(
            issue for issue in self._state.issues if issue.block_id == block_id
        )

    def _find_block(self, block_id: str) -> tuple[LaneId, int, PromptBlock] | None:
        for lane in self._state.definition.lanes:
            for index, block in enumerate(lane.blocks):
                if block.id == block_id:
                    return lane.id, index, block
        return None

    def _block_id_from_widget_id(self, widget_id: str | None, prefix: str) -> str:
        if widget_id is None or not widget_id.startswith(prefix):
            raise ValueError(f"Unexpected block control id: {widget_id!r}")
        token = widget_id.removeprefix(prefix)
        if token.startswith("encoded-"):
            return bytes.fromhex(token.removeprefix("encoded-")).decode("utf-8")
        return token

    @on(Input.Changed, ".prompt-block-title")
    async def _title_changed(self, event: Input.Changed) -> None:
        event.stop()
        block_id = self._block_id_from_widget_id(event.input.id, "prompt-block-title-")
        await self._change_field(block_id, "title", event.value)

    @on(Input.Changed, ".prompt-block-xml-tag")
    async def _xml_tag_changed(self, event: Input.Changed) -> None:
        event.stop()
        block_id = self._block_id_from_widget_id(
            event.input.id, "prompt-block-xml-tag-"
        )
        await self._change_field(block_id, "xml_tag", event.value)

    @on(TextArea.Changed, ".prompt-block-content")
    async def _content_changed(self, event: TextArea.Changed) -> None:
        event.stop()
        block_id = self._block_id_from_widget_id(
            event.text_area.id, "prompt-block-content-"
        )
        await self._change_field(block_id, "content", event.text_area.text)

    @on(Select.Changed, ".prompt-block-syntax")
    async def _syntax_changed(self, event: Select.Changed) -> None:
        event.stop()
        value = event.value
        if value not in {"freeform", "markdown", "xml"}:
            return
        block_id = self._block_id_from_widget_id(
            event.select.id, "prompt-block-syntax-"
        )
        await self._change_field(block_id, "syntax", str(value))

    async def _change_field(self, block_id: str, field: BlockField, value: str) -> None:
        located = self._find_block(block_id)
        if located is None:
            return
        current = getattr(located[2], field)
        if field == "xml_tag" and current is None:
            current = ""
        if current == value:
            return
        state = update_block(self._state, block_id, **{field: value})
        await self.replace_block_state(block_id, state)
        self.post_message(self.BlockFieldChanged(block_id, field, value, state))

    @on(Checkbox.Changed)
    def _lane_selection_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id not in {
            "prompt-editor-apply-system",
            "prompt-editor-apply-user",
        }:
            return
        event.stop()
        self._sync_footer()

    @on(Button.Pressed)
    async def _button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("prompt-"):
            return
        event.stop()
        if button_id == "prompt-editor-back":
            self.post_message(self.BackRequested())
            return
        if button_id == "prompt-editor-save-prompt":
            self._request_save("prompt")
            return
        if button_id == "prompt-editor-save-recipe":
            self._request_save("recipe")
            return
        if button_id == "prompt-editor-update-original":
            self._request_update()
            return
        if button_id == "prompt-editor-apply":
            self._request_apply()
            return
        if button_id.startswith("prompt-lane-add-"):
            lane_id = button_id.removeprefix("prompt-lane-add-")
            if lane_id in {"system", "user"}:
                await self._add_lane_block(lane_id)
            return

        action_prefixes: tuple[tuple[str, BlockAction, int | None], ...] = (
            ("prompt-block-move-up-", "move_up", -1),
            ("prompt-block-move-down-", "move_down", 1),
            ("prompt-block-duplicate-", "duplicate", None),
            ("prompt-block-delete-", "delete", None),
        )
        for prefix, action, direction in action_prefixes:
            if not button_id.startswith(prefix):
                continue
            block_id = self._block_id_from_widget_id(button_id, prefix)
            located = self._find_block(block_id)
            if located is None:
                return
            lane_id = located[0]
            if action in {"move_up", "move_down"}:
                assert direction is not None
                state = move_block(self._state, block_id, direction)
            elif action == "duplicate":
                state = duplicate_block(self._state, block_id)
            else:
                state = delete_block(self._state, block_id)
            if state is not self._state:
                await self.replace_block_state(block_id, state)
                self.post_message(
                    self.BlockActionRequested(
                        action,
                        block_id=block_id,
                        lane_id=lane_id,
                        state=state,
                    )
                )
            return

    async def _add_lane_block(self, lane_id: LaneId) -> None:
        previous_ids = {
            block.id for lane in self._state.definition.lanes for block in lane.blocks
        }
        state = add_block(self._state, lane_id)
        new_id = next(
            block.id
            for lane in state.definition.lanes
            for block in lane.blocks
            if block.id not in previous_ids
        )
        await self.replace_block_state(new_id, state)
        self.post_message(
            self.BlockActionRequested(
                "add",
                block_id=new_id,
                lane_id=lane_id,
                state=state,
            )
        )

    def _request_save(self, artifact_type: Literal["prompt", "recipe"]) -> None:
        if self._state.issues:
            self._focus_first_error()
            return
        if artifact_type == "recipe" and not self._state.can_save_as_recipe:
            return
        if artifact_type == "prompt":
            self.post_message(self.SaveAsPromptRequested(self._state))
        else:
            self.post_message(self.SaveAsRecipeRequested(self._state))

    def _request_update(self) -> None:
        if self._state.issues:
            self._focus_first_error()
            return
        if self._can_update_original:
            self.post_message(self.UpdateOriginalRequested(self._state))

    def _request_apply(self) -> None:
        if self._state.issues:
            self._focus_first_error()
            return
        system = self.query_one("#prompt-editor-apply-system", Checkbox)
        user = self.query_one("#prompt-editor-apply-user", Checkbox)
        apply_system = bool(
            self._allow_apply_system and system.value and self._state.compiled_system
        )
        apply_user = bool(user.value and self._state.compiled_user)
        if not (apply_system or apply_user):
            self._sync_footer()
            return
        self.post_message(
            self.ApplyRequested(
                apply_system=apply_system,
                apply_user=apply_user,
                system_prompt=self._state.compiled_system if apply_system else None,
                user_prompt=self._state.compiled_user if apply_user else None,
                state=self._state,
            )
        )

    def _focus_first_error(self) -> None:
        if not self._state.issues:
            return
        issue = self._state.issues[0]
        token = self.widget_token_for_block_id(issue.block_id)
        field = (
            issue.field if issue.field in {"title", "xml_tag", "content"} else "title"
        )
        selector = f"#prompt-block-{field.replace('_', '-')}-{token}"
        target = self.query_one(selector)
        target.focus()
        self.scroll_to_widget(target, animate=False)

    def action_apply(self) -> None:
        """Attempt Apply and focus the first blocking field when invalid."""
        self._request_apply()

    def action_save_prompt(self) -> None:
        """Attempt Save as Prompt and focus the first blocking field."""
        self._request_save("prompt")
