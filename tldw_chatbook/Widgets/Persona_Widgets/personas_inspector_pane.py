"""Selected-item inspector pane for the Personas workbench."""

from __future__ import annotations

import re

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import QueryError
from textual.widgets import Button, Checkbox, ListItem, ListView, Static

from ..Console.console_image_viewer_modal import ClickableAvatarBox

from .personas_messages import PersonaBuddyActionRequested
from .personas_pane_messages import ConversationRowSelected

_UNSAVED_TOOLTIP = "Save before using this action; the selection has unsaved edits."

#: F-037: every disabled action explains itself. The screen hides the whole
#: action stack pre-selection (F-031), but the pane keeps the disabled+reason
#: contract for every state these buttons can render in.
_NO_SELECTION_EXPORT_TOOLTIP = "Select an item to export."
_NO_SELECTION_DELETE_TOOLTIP = "Select an item to delete."

#: The one plain line a no-selection inspector shows (F-031): intent-first
#: guidance instead of a wall of disabled controls and a false
#: "Validation: OK". Doubles as the Console readiness line pre-selection.
_NO_SELECTION_GUIDANCE = "Pick a character or persona to start chatting."

_ID_SAFE = re.compile(r"[^a-zA-Z0-9_-]")

# Kind-applicability for actions that can never apply to some selections
# (task-443): dictionaries/lore have no Console handoff and no card export,
# and personas have no PNG card. "Never applies" is a rendering decision
# (this button does not belong on this selection at all) and is kept
# separate from "applies but is currently blocked" (unsaved edits, provider
# readiness, no selection yet), which stays owned by
# set_console_actions_enabled/_apply_action_state's disabled+tooltip logic
# below. Before a kind is known (no selection) the per-button display flags
# still reset to visible, but the whole action stack is hidden behind the
# no-selection guidance line (F-031), so the flags only take effect once a
# selection reveals the stack again.
_CONSOLE_ACTION_APPLICABLE_KINDS = {"character", "persona"}
_EXPORT_JSON_APPLICABLE_KINDS = {"character", "persona"}
_EXPORT_PNG_APPLICABLE_KINDS = {"character"}

# Portrait thumb box in character cells. Must stay in sync with
# #personas-inspector-avatar-thumb's CSS max-width/max-height below and with
# AVATAR_THUMB_COLS/AVATAR_THUMB_LINES in personas_screen.py - change one,
# change all three.
_THUMB_BOX_COLS = 24
_THUMB_BOX_LINES = 10


class PersonasInspectorPane(Vertical):
    """Identity, validation, conversations, readiness, and actions."""

    # Structure only: colors come from the app stylesheet. The conversations
    # list is CAPPED (scrolls past 10 rows) so the Readiness section and the
    # action buttons below it are always visible when the pane renders.
    # Rows are ListItems in a ListView (keyboard-first, Notes idiom).
    BUNDLED_CSS = """
    /* Portrait box for the selected character. Sized like the editor's
       thumbnail rather than the 80x40 transcript image box: this is a single
       always-visible preview, and the rail is narrow. `height: auto` keeps the
       box from reserving space for selections that have no portrait. No
       horizontal padding: the mosaic renderable is baked at exactly
       AVATAR_THUMB_COLS columns, and padding used to shrink the content box
       below that, folding every line into a black continuation stripe
       (task-3793). */
    PersonasInspectorPane #personas-inspector-avatar-thumb {
        height: auto;
        max-width: 24;
        max-height: 10;
    }

    PersonasInspectorPane #personas-conversations-list {
        height: auto;
        max-height: 10;
    }

    PersonasInspectorPane #personas-readiness-console {
        width: 100%;
        min-width: 0;
        height: auto;
        text-wrap: wrap;
    }

    PersonasInspectorPane .personas-conversation-row {
        width: 100%;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    PersonasInspectorPane .personas-conversation-row Static {
        width: 100%;
        height: 1;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    PersonasInspectorPane #personas-inspector-actions {
        height: auto;
    }

    PersonasInspectorPane #personas-inspector-actions Button {
        width: 100%;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    PersonasInspectorPane #personas-export-include-tts {
        width: 100%;
        height: auto;
    }

    /* F-041: a disabled checkbox must still READ as a checkbox. Textual's
       base *:disabled:can-focus rule dims it to 0.7 and the stock toggle
       glyph box is panel-on-panel (a dark gap in the rail). Keep the label
       dimmed per the ds disabled idiom, restore full opacity, and give the
       glyph box a visible surface. ($text-disabled/$surface are the theme
       variables the ds-text-disabled/ds-surface-raised tokens alias; widget
       DEFAULT_CSS cannot see bundle-scoped ds-* names.) */
    PersonasInspectorPane #personas-export-include-tts:disabled {
        opacity: 100%;
        color: $text-disabled;
    }

    PersonasInspectorPane #personas-export-include-tts:disabled .toggle--button {
        background: $surface;
        color: $text;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._has_selection = False
        self._is_unsaved = False
        self._selected_kind: str | None = None
        self._console_actions_enabled = False
        self._console_action_block_reason = "select an item"
        self._provider_block_reason: str | None = None
        self._conversation_lookup: dict[str, str] = {}
        self._tts_export_available = False
        self._buddy_source: str | None = None
        self._buddy_persona_id: str | None = None
        self._buddy_revision: int | None = None
        self._buddy_active = False
        self._buddy_profile_current = False
        # F-040: marked library rows drive bulk Delete/Export JSON affordances.
        self._marked_count = 0

    def set_marked_count(self, count: int) -> None:
        """Bulk-mark awareness for Delete/Export (F-040).

        With marks active, Delete and Export JSON target the marked set
        (their tooltips say so) and Export PNG - a single-card format - is
        disabled with a reason. Zero restores the selection-owned gates.
        """
        self._marked_count = max(0, int(count))
        self._apply_action_state()

    def compose(self) -> ComposeResult:
        """Compose the Inspector pane summary, readiness, and actions.

        Returns:
            Textual compose result for the Inspector pane.
        """
        with Horizontal(classes="console-rail-header"):
            title = Static(
                "Inspector",
                classes="destination-section personas-column-title console-rail-title",
            )
            title.styles.width = "1fr"
            yield title
            collapse_button = Button(
                ">",
                id="personas-inspector-rail-collapse",
                classes="console-rail-collapse-button",
                compact=True,
            )
            collapse_button.tooltip = "Collapse Inspector rail"
            yield collapse_button
        yield Static("Selected: none", id="personas-selected-name")
        yield Static("Type: -", id="personas-selected-kind")
        # Portrait of the selected character. A roleplay user identifies a
        # character by its picture at least as much as by its name, and the
        # inspector previously showed every attribute EXCEPT the portrait.
        yield ClickableAvatarBox(id="personas-inspector-avatar-thumb")
        # F-031: everything below the portrait is hidden until there is a
        # selection - pre-selection the inspector is just the summary lines
        # plus one plain guidance line (the readiness Static below), not a
        # false "Validation: OK", dangling section headers, and dead buttons.
        validation = Static("Validation: OK", id="personas-validation-summary")
        validation.display = False
        yield validation
        conversations_header = Static(
            "Conversations",
            id="personas-conversations-header",
            classes="destination-section",
        )
        conversations_header.display = False
        yield conversations_header
        conversations_list = ListView(id="personas-conversations-list")
        conversations_list.display = False
        yield conversations_list
        readiness_header = Static(
            "Readiness",
            id="personas-readiness-header",
            classes="destination-section",
        )
        readiness_header.display = False
        yield readiness_header
        yield Static(_NO_SELECTION_GUIDANCE, id="personas-readiness-console")
        actions = Vertical(id="personas-inspector-actions")
        actions.display = False
        with actions:
            # F-032: one primary Console CTA named by intent (Chat now =
            # begin immediately) plus one secondary (Send to Console draft =
            # stage the card for the next Console draft). Ids, handlers, and
            # the per-intent gating (task-523: Chat now also needs a ready
            # provider) are unchanged - only labels, order, and emphasis.
            yield Button(
                "Chat now",
                id="personas-start-chat",
                disabled=True,
                classes="console-action-primary",
                tooltip=_NO_SELECTION_GUIDANCE,
            )
            yield Button(
                "Send to Console draft",
                id="personas-attach-to-console",
                disabled=True,
                classes="console-action-secondary",
                tooltip=_NO_SELECTION_GUIDANCE,
            )
            yield Button(
                "Use for Buddy",
                id="personas-buddy-use",
                disabled=True,
                classes="console-action-secondary persona-buddy-action",
            )
            yield Button(
                "Show Buddy",
                id="personas-buddy-show",
                disabled=True,
                classes="console-action-subdued persona-buddy-action",
            )
            yield Button(
                "Close Buddy",
                id="personas-buddy-close",
                disabled=True,
                classes="console-action-subdued persona-buddy-action",
            )
            yield Button(
                "Disable Buddy",
                id="personas-buddy-disable",
                disabled=True,
                classes="console-action-subdued persona-buddy-action",
            )
            tts_checkbox = Checkbox(
                "Include assigned voice profile",
                id="personas-export-include-tts",
                value=False,
                disabled=True,
            )
            # task-2233: starts hidden (no profile assigned yet);
            # _apply_action_state reveals it once an assignment is known.
            tts_checkbox.display = False
            yield tts_checkbox
            yield Button(
                "Export JSON",
                id="personas-export-json",
                disabled=True,
                classes="console-action-subdued",
                tooltip=_NO_SELECTION_EXPORT_TOOLTIP,
            )
            yield Button(
                "Export PNG",
                id="personas-export-png",
                disabled=True,
                classes="console-action-subdued",
                tooltip=_NO_SELECTION_EXPORT_TOOLTIP,
            )
            yield Button(
                "Delete",
                id="personas-delete",
                disabled=True,
                classes="console-action-subdued personas-destructive",
                tooltip=_NO_SELECTION_DELETE_TOOLTIP,
            )

    def show_selection(
        self,
        *,
        name: str,
        kind: str,
        source: str | None = None,
        entity_id: str | None = None,
        revision: int | None = None,
        active: bool = False,
        profile_current: bool = False,
    ) -> None:
        """Reflect the selected library item in the inspector summary.

        Args:
            name: The selected item's display name.
            kind: The selection kind (``character``/``persona``/
                ``dictionary``/``lore``) -- drives which actions render
                (task-443 kind-aware visibility).
        """
        self._has_selection = True
        self._selected_kind = kind
        self._buddy_source = source
        self._buddy_persona_id = entity_id
        self._buddy_revision = revision
        self._buddy_active = active is True
        self._buddy_profile_current = profile_current is True
        self._tts_export_available = False
        self.query_one("#personas-export-include-tts", Checkbox).value = False
        self.query_one("#personas-selected-name", Static).update(f"Selected: {name}")
        self.query_one("#personas-selected-kind", Static).update(f"Type: {kind}")
        self._apply_action_state()

    async def clear_selection(self) -> None:
        self._has_selection = False
        self._is_unsaved = False
        self._selected_kind = None
        self._buddy_source = None
        self._buddy_persona_id = None
        self._buddy_revision = None
        self._buddy_active = False
        self._buddy_profile_current = False
        self._tts_export_available = False
        self.query_one("#personas-export-include-tts", Checkbox).value = False
        self.set_console_actions_enabled(False, reason="select an item")
        self.query_one("#personas-selected-name", Static).update("Selected: none")
        self.query_one("#personas-selected-kind", Static).update("Type: -")
        await self.show_conversations(())
        self.show_validation(())
        self._apply_action_state()

    def set_unsaved(self, is_unsaved: bool) -> None:
        self._is_unsaved = is_unsaved
        self._apply_action_state()

    def set_tts_export_available(self, available: bool) -> None:
        """Expose explicit inclusion only when the selected card has a profile."""

        self._tts_export_available = bool(available)
        if not self._tts_export_available:
            self.query_one("#personas-export-include-tts", Checkbox).value = False
        self._apply_action_state()

    @property
    def include_tts_profile_in_export(self) -> bool:
        """Return the current explicit opt-in; assignment presence is insufficient."""

        checkbox = self.query_one("#personas-export-include-tts", Checkbox)
        return (
            self._selected_kind == "character"
            and self._tts_export_available
            and not checkbox.disabled
            and checkbox.value
        )

    def set_console_actions_enabled(
        self,
        enabled: bool,
        *,
        reason: str | None = None,
        provider_block_reason: str | None = None,
    ) -> None:
        """Set Chat-now/Send-draft availability from the screen-owned Console gate.

        Selection, export, and delete state stay local to the inspector, but
        Console action availability must be pushed by ``PersonasScreen`` so
        the visible buttons, readiness copy, and shortcuts cannot diverge.

        Args:
            enabled: Whether Console actions are currently available.
            reason: Optional user-facing reason shown when actions are blocked.
            provider_block_reason: Optional blocker naming why the provider a
                Chat now/Send to Console draft handoff session would resolve
                is not ready (task-440). Per-intent gating (task-523): when
                set (and ``enabled`` is True) it replaces the "Ready to chat
                in Console." copy with "Chat now blocked: ..." and DISABLES
                Chat now - which needs an immediate provider reply - while
                Send to Console draft stays enabled: it only stages context,
                so its send is deferred and the user can fix the provider
                before sending. Ignored when ``enabled`` is False - the
                selection/unsaved reason already owns the copy and both
                buttons are disabled by the gate.
        """
        self._console_actions_enabled = bool(enabled)
        self._console_action_block_reason = "" if enabled else (reason or "unavailable")
        self._provider_block_reason = provider_block_reason if enabled else None
        self._apply_action_state()

    def show_validation(self, errors: tuple[str, ...]) -> None:
        summary = self.query_one("#personas-validation-summary", Static)
        if errors:
            summary.update("Validation errors:\n" + "\n".join(errors))
        else:
            summary.update("Validation: OK")

    def show_validation_editing(self) -> None:
        """Editing-session state: the editor footer owns the error detail,
        so the inspector line must not claim "OK" while an editor is open."""
        self.query_one("#personas-validation-summary", Static).update(
            "Validation: editing..."
        )

    async def show_conversations_loading(self) -> None:
        """Show a loading placeholder while the listing worker runs."""
        await self._show_conversations_placeholder("Loading conversations...")

    async def _show_conversations_placeholder(self, text: str) -> None:
        """Replace the rows with one disabled, non-selectable status line."""
        list_view = self.query_one("#personas-conversations-list", ListView)
        await list_view.clear()
        self._conversation_lookup = {}
        await list_view.extend(
            [
                ListItem(
                    Static(text, markup=False),
                    classes="personas-conversations-placeholder",
                    disabled=True,
                )
            ]
        )

    async def show_conversations(
        self,
        rows: tuple[tuple[str, str], ...],
        *,
        empty_copy: str | None = None,
    ) -> None:
        """Render (conversation_id, title) rows.

        An empty ``rows`` tuple clears the panel silently unless
        ``empty_copy`` is given, in which case that copy renders as a
        disabled placeholder (the library empty-state idiom).
        """
        list_view = self.query_one("#personas-conversations-list", ListView)
        await list_view.clear()
        self._conversation_lookup = {}
        if not rows and empty_copy:
            await self._show_conversations_placeholder(empty_copy)
            return
        items: list[ListItem] = []
        seen: set[str] = set()
        for conversation_id, title in rows:
            dom_id = (
                f"personas-conversation-row-{_ID_SAFE.sub('-', str(conversation_id))}"
            )
            if dom_id in seen:
                suffix = 2
                while f"{dom_id}-{suffix}" in seen:
                    suffix += 1
                dom_id = f"{dom_id}-{suffix}"
            seen.add(dom_id)
            self._conversation_lookup[dom_id] = conversation_id
            items.append(
                ListItem(
                    Static(title, markup=False),
                    id=dom_id,
                    classes="personas-conversation-row console-action-subdued",
                )
            )
        if items:
            await list_view.extend(items)

    def on_mount(self) -> None:
        """Replay any state pushed before the composed children existed.

        task-2727: `PersonasScreen._load_after_mount` runs as a worker and can
        push console-action state while this pane's children are still
        mounting; `_apply_action_state` defers quietly in that window, and
        this replay (after the first refresh, so the children are queryable)
        makes sure nothing pushed early is dropped.
        """
        self.call_after_refresh(self._apply_action_state)

    def _apply_action_state(self) -> None:
        selected = self._has_selection
        unsaved = self._is_unsaved
        kind = self._selected_kind
        try:
            self.query_one("#personas-validation-summary", Static)
        except QueryError:
            # Children not composed yet (pre-mount push) — the on_mount
            # replay applies the retained state once they exist (task-2727).
            return
        # F-031: pre-selection the inspector is one guidance line only. The
        # section chrome (Validation, Conversations, Readiness header) and
        # the action stack stay hidden until there is something to act on;
        # per-button kind gating below is unchanged for when they render.
        self.query_one("#personas-validation-summary", Static).display = selected
        self.query_one("#personas-readiness-header", Static).display = selected
        self.query_one("#personas-inspector-actions", Vertical).display = selected
        # F-036: only characters have saved conversations - the section hides
        # for persona/dictionary/lore selections (the task-443 kind idiom)
        # instead of dangling a header over an empty list.
        conversations_visible = selected and kind == "character"
        self.query_one(
            "#personas-conversations-header", Static
        ).display = conversations_visible
        self.query_one(
            "#personas-conversations-list", ListView
        ).display = conversations_visible
        readiness = self.query_one("#personas-readiness-console", Static)
        # F-032: readiness speaks in intent (what to do next), not app
        # topology ("Console blocked: ..."). The per-intent gating is
        # unchanged - only the copy moved.
        if not selected:
            readiness.update(_NO_SELECTION_GUIDANCE)
        elif not self._console_actions_enabled:
            if unsaved:
                readiness.update("Save or discard your edits to chat in Console.")
            elif kind not in _CONSOLE_ACTION_APPLICABLE_KINDS:
                readiness.update("Console chat is for characters and personas.")
            else:
                reason = self._console_action_block_reason or "unavailable"
                # task-2232: the gate closed blocks BOTH CTAs, so the copy
                # names the pair (one vocabulary everywhere).
                readiness.update(
                    f"Chat now and Send to Console draft blocked: {reason}"
                )
        elif self._provider_block_reason:
            # Per-intent (task-523): Send to Console draft stays available;
            # only Chat now is blocked because it needs an immediate reply
            # from the provider.
            readiness.update(f"Chat now blocked: {self._provider_block_reason}")
        else:
            readiness.update("Ready to chat in Console.")
        export_enabled = selected and not unsaved
        # F-037: every disabled action carries a reason - unsaved edits when
        # that is the blocker, the select-first reason pre-selection.
        if selected and unsaved:
            export_tooltip: str | None = _UNSAVED_TOOLTIP
        elif not selected:
            export_tooltip = _NO_SELECTION_EXPORT_TOOLTIP
        else:
            export_tooltip = None
        # Send-to-draft tooltip only surfaces when the selection gate itself
        # is closed; the copy matches the readiness line's intent language.
        attach_tooltip = None
        if not self._console_actions_enabled:
            if not selected:
                attach_tooltip = _NO_SELECTION_GUIDANCE
            elif unsaved:
                attach_tooltip = _UNSAVED_TOOLTIP
            else:
                attach_tooltip = (
                    "Chat now and Send to Console draft blocked: "
                    f"{self._console_action_block_reason}"
                )
        # Kind gates rendering (task-443); readiness/unsaved/provider-readiness
        # gate the disabled+tooltip state of whatever is rendered (see the
        # module-level constants).
        console_applies = kind is None or kind in _CONSOLE_ACTION_APPLICABLE_KINDS
        export_json_applies = kind is None or kind in _EXPORT_JSON_APPLICABLE_KINDS
        export_png_applies = kind is None or kind in _EXPORT_PNG_APPLICABLE_KINDS
        tts_checkbox = self.query_one("#personas-export-include-tts", Checkbox)
        # task-2233: hidden outright unless the selection actually HAS a
        # voice profile to include - a permanently-disabled "not applicable"
        # checkbox read as an unreadable dark smear right under the primary
        # CTA. When a profile IS assigned, the enabled/disabled-with-reason
        # gating below (and the F-041 legibility CSS) covers the shown case.
        tts_checkbox.display = (
            kind is None or kind == "character"
        ) and self._tts_export_available
        tts_checkbox.disabled = not (
            export_enabled and kind == "character" and self._tts_export_available
        )
        tts_checkbox.tooltip = (
            export_tooltip
            if export_tooltip is not None
            else (
                None
                if self._tts_export_available
                else "Assign a voice profile before including it."
            )
        )
        # Send to Console draft: the selection gate only (staging context
        # defers the reply).
        attach_btn = self.query_one("#personas-attach-to-console", Button)
        attach_btn.display = console_applies
        attach_btn.disabled = not self._console_actions_enabled
        attach_btn.tooltip = attach_tooltip
        # Chat now: selection AND a ready handoff provider (task-523) -- it
        # needs an immediate reply, so an unready provider disables it while
        # Send to Console draft stays available.
        start_btn = self.query_one("#personas-start-chat", Button)
        start_btn.display = console_applies
        start_btn.disabled = (not self._console_actions_enabled) or bool(
            self._provider_block_reason
        )
        if not self._console_actions_enabled:
            start_btn.tooltip = attach_tooltip
        elif self._provider_block_reason:
            start_btn.tooltip = f"Chat now blocked: {self._provider_block_reason}"
        else:
            start_btn.tooltip = None
        # F-040: an active mark set retargets Delete/Export JSON at the
        # marked rows (tooltips say so) and sidesteps the single-selection
        # unsaved gate (bulk actions discard/ignore edit state by design,
        # like single Delete). Export PNG stays single-card.
        marked = self._marked_count
        json_button = self.query_one("#personas-export-json", Button)
        json_button.display = export_json_applies
        json_button.disabled = (not export_enabled) and marked == 0
        json_button.tooltip = (
            f"Export the {marked} marked items as JSON." if marked else export_tooltip
        )
        png_button = self.query_one("#personas-export-png", Button)
        png_button.display = export_png_applies
        png_button.disabled = marked > 0 or not (export_enabled and kind == "character")
        png_button.tooltip = "Bulk export is JSON only." if marked else export_tooltip
        delete_button = self.query_one("#personas-delete", Button)
        delete_button.disabled = (not selected) and marked == 0
        delete_button.tooltip = (
            f"Delete the {marked} marked items."
            if marked
            else (None if selected else _NO_SELECTION_DELETE_TOOLTIP)
        )

        buddy_applies = selected and kind == "persona"
        buddy_eligible = (
            buddy_applies
            and not unsaved
            and self._buddy_profile_current
            and self._buddy_source == "local"
            and bool(self._buddy_persona_id)
            and type(self._buddy_revision) is int
            and self._buddy_revision >= 1
            and self._buddy_active
        )
        if self._buddy_source == "server":
            buddy_tooltip = "Save a local copy first"
        elif unsaved:
            buddy_tooltip = _UNSAVED_TOOLTIP
        elif not self._buddy_profile_current:
            buddy_tooltip = "Persona details are unavailable. Refresh and try again."
        elif not self._buddy_active:
            buddy_tooltip = "Activate this Persona first."
        else:
            buddy_tooltip = "Select a saved local Persona."
        for button in self.query(".persona-buddy-action").results(Button):
            button.display = buddy_applies
            button.disabled = not buddy_eligible
            button.tooltip = None if buddy_eligible else buddy_tooltip

    @on(Button.Pressed, ".persona-buddy-action")
    def _buddy_action_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        actions = {
            "personas-buddy-use": "use",
            "personas-buddy-show": "show",
            "personas-buddy-close": "close",
            "personas-buddy-disable": "disable",
        }
        action = actions.get(str(event.button.id or ""))
        if (
            action is None
            or self._buddy_source not in {"local", "server"}
            or not self._buddy_persona_id
            or type(self._buddy_revision) is not int
        ):
            return
        self.post_message(
            PersonaBuddyActionRequested(
                action=action,
                source=self._buddy_source,
                persona_id=self._buddy_persona_id,
                revision=self._buddy_revision,
            )
        )

    def set_avatar_thumbnail(self, renderable: object | None) -> None:
        """Mount a prepared portrait renderable, or clear the box.

        Mirrors ``PersonasCharacterEditorWidget.set_avatar_thumbnail``: the
        screen owns decoding off-thread via ``ConsoleImageRenderCache`` and
        passes a finished renderable here. A rich renderable (e.g.
        ``rich_pixels.Pixels``) mounts inside a ``Static``; a Textual widget
        (e.g. a ``textual_image`` graphics ``Image``) mounts directly.

        Args:
            renderable: Prepared renderable to display, or ``None`` to clear
                (selections with no portrait, and non-character kinds).
        """
        try:
            holder = self.query_one("#personas-inspector-avatar-thumb", Container)
        except Exception:
            return
        holder.remove_children()
        if renderable is None:
            return
        from textual.widget import Widget as _W

        if isinstance(renderable, _W):
            holder.mount(renderable)
            return
        from ...Utils.mosaic_render import explicit_cell_size

        # Explicit cell size from the baked renderable's grid: the Static
        # default width: 100% folds a full-width mosaic inside any narrower
        # (or padded) box, painting black continuation stripes (task-3793);
        # an explicit size degrades a future width mismatch to a crop.
        thumb = Static(renderable)
        grid_size = explicit_cell_size(renderable)
        if grid_size is not None:
            thumb.styles.width, thumb.styles.height = grid_size
        else:
            # Per explicit_cell_size's documented contract, fall back to the
            # box dimensions when the grid can't be read (e.g. rich_pixels
            # Pixels, which is baked for the box anyway) - same fallback as
            # ChatScreen._build_character_avatar_widget.
            thumb.styles.width = _THUMB_BOX_COLS
            thumb.styles.height = _THUMB_BOX_LINES
        holder.mount(thumb)

    @on(ListView.Selected, "#personas-conversations-list")
    def _conversation_selected(self, event: ListView.Selected) -> None:
        event.stop()
        conversation_id = self._conversation_lookup.get(str(event.item.id or ""))
        if conversation_id is not None:
            self.post_message(ConversationRowSelected(conversation_id))
