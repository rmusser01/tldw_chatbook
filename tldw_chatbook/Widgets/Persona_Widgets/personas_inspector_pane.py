"""Selected-item inspector pane for the Personas workbench."""

from __future__ import annotations

import re

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, ListItem, ListView, Static

from .personas_pane_messages import ConversationRowSelected

_UNSAVED_TOOLTIP = "Save before using this action; the selection has unsaved edits."

_ID_SAFE = re.compile(r"[^a-zA-Z0-9_-]")

# Kind-applicability for actions that can never apply to some selections
# (task-443): dictionaries/lore have no Console handoff and no card export,
# and personas have no PNG card. "Never applies" is a rendering decision
# (this button does not belong on this selection at all) and is kept
# separate from "applies but is currently blocked" (unsaved edits, provider
# readiness, no selection yet), which stays owned by
# set_console_actions_enabled/_apply_action_state's disabled+tooltip logic
# below. Before a kind is known (no selection) every action still renders,
# matching the pre-selection "select an item" baseline.
_CONSOLE_ACTION_APPLICABLE_KINDS = {"character", "persona_profile"}
_EXPORT_JSON_APPLICABLE_KINDS = {"character", "persona_profile"}
_EXPORT_PNG_APPLICABLE_KINDS = {"character"}


class PersonasInspectorPane(Vertical):
    """Identity, validation, conversations, readiness, and actions."""

    # Structure only: colors come from the app stylesheet. The conversations
    # list is CAPPED (scrolls past 10 rows) so the Readiness section and the
    # action buttons below it are always visible when the pane renders.
    # Rows are ListItems in a ListView (keyboard-first, Notes idiom).
    DEFAULT_CSS = """
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
        yield Static("Validation: OK", id="personas-validation-summary")
        yield Static("Conversations", classes="destination-section")
        yield ListView(id="personas-conversations-list")
        yield Static("Readiness", classes="destination-section")
        yield Static("Console blocked: select an item", id="personas-readiness-console")
        with Vertical(id="personas-inspector-actions"):
            yield Button(
                "Attach to Console",
                id="personas-attach-to-console",
                disabled=True,
                classes="console-action-secondary",
            )
            yield Button(
                "Start Chat",
                id="personas-start-chat",
                disabled=True,
                classes="console-action-secondary",
            )
            yield Button(
                "Export JSON",
                id="personas-export-json",
                disabled=True,
                classes="console-action-subdued",
            )
            yield Button(
                "Export PNG",
                id="personas-export-png",
                disabled=True,
                classes="console-action-subdued",
            )
            yield Button(
                "Delete",
                id="personas-delete",
                disabled=True,
                classes="console-action-subdued personas-destructive",
            )

    def show_selection(self, *, name: str, kind: str) -> None:
        """Reflect the selected library item in the inspector summary.

        Args:
            name: The selected item's display name.
            kind: The selection kind (``character``/``persona_profile``/
                ``dictionary``/``lore``) -- drives which actions render
                (task-443 kind-aware visibility).
        """
        self._has_selection = True
        self._selected_kind = kind
        self.query_one("#personas-selected-name", Static).update(f"Selected: {name}")
        self.query_one("#personas-selected-kind", Static).update(f"Type: {kind}")
        self._apply_action_state()

    async def clear_selection(self) -> None:
        self._has_selection = False
        self._is_unsaved = False
        self._selected_kind = None
        self.set_console_actions_enabled(False, reason="select an item")
        self.query_one("#personas-selected-name", Static).update("Selected: none")
        self.query_one("#personas-selected-kind", Static).update("Type: -")
        await self.show_conversations(())
        self.show_validation(())
        self._apply_action_state()

    def set_unsaved(self, is_unsaved: bool) -> None:
        self._is_unsaved = is_unsaved
        self._apply_action_state()

    def set_console_actions_enabled(
        self,
        enabled: bool,
        *,
        reason: str | None = None,
        provider_block_reason: str | None = None,
    ) -> None:
        """Set Attach/Start availability from the screen-owned Console gate.

        Selection, export, and delete state stay local to the inspector, but
        Console action availability must be pushed by ``PersonasScreen`` so
        the visible buttons, readiness copy, and shortcuts cannot diverge.

        Args:
            enabled: Whether Console actions are currently available.
            reason: Optional user-facing reason shown when actions are blocked.
            provider_block_reason: Optional blocker naming why the provider a
                Start Chat/Attach Console handoff session would resolve is not
                ready (task-440). Per-intent gating (task-523): when set (and
                ``enabled`` is True) it replaces the "Console ready" copy with
                "Start Chat blocked: ..." and DISABLES Start Chat - which needs
                an immediate provider reply - while Attach stays enabled: Attach
                only stages context, so its send is deferred and the user can
                fix the provider before sending. Ignored when ``enabled`` is
                False - the selection/unsaved reason already owns the copy and
                both buttons are disabled by the gate.
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

    def _apply_action_state(self) -> None:
        selected = self._has_selection
        unsaved = self._is_unsaved
        kind = self._selected_kind
        readiness = self.query_one("#personas-readiness-console", Static)
        if not self._console_actions_enabled:
            reason = self._console_action_block_reason or "unavailable"
            readiness.update(f"Console blocked: {reason}")
        elif self._provider_block_reason:
            # Per-intent (task-523): Attach stays available; only Start Chat is
            # blocked because it needs an immediate reply from the provider.
            readiness.update(f"Start Chat blocked: {self._provider_block_reason}")
        else:
            readiness.update("Console ready")
        export_enabled = selected and not unsaved
        export_tooltip = _UNSAVED_TOOLTIP if (selected and unsaved) else None
        # Attach: the selection gate only (staging context defers the reply).
        attach_tooltip = None
        if not self._console_actions_enabled:
            attach_tooltip = (
                _UNSAVED_TOOLTIP
                if selected and unsaved
                else f"Console action blocked: {self._console_action_block_reason}"
            )
        # Kind gates rendering (task-443); readiness/unsaved/provider-readiness
        # gate the disabled+tooltip state of whatever is rendered (see the
        # module-level constants).
        console_applies = kind is None or kind in _CONSOLE_ACTION_APPLICABLE_KINDS
        export_json_applies = kind is None or kind in _EXPORT_JSON_APPLICABLE_KINDS
        export_png_applies = kind is None or kind in _EXPORT_PNG_APPLICABLE_KINDS
        # Attach: the selection gate only (staging context defers the reply).
        attach_btn = self.query_one("#personas-attach-to-console", Button)
        attach_btn.display = console_applies
        attach_btn.disabled = not self._console_actions_enabled
        attach_btn.tooltip = attach_tooltip
        # Start Chat: selection AND a ready handoff provider (task-523) -- it
        # needs an immediate reply, so an unready provider disables it while
        # Attach stays available.
        start_btn = self.query_one("#personas-start-chat", Button)
        start_btn.display = console_applies
        start_btn.disabled = (not self._console_actions_enabled) or bool(
            self._provider_block_reason
        )
        if not self._console_actions_enabled:
            start_btn.tooltip = attach_tooltip
        elif self._provider_block_reason:
            start_btn.tooltip = f"Start Chat blocked: {self._provider_block_reason}"
        else:
            start_btn.tooltip = None
        json_button = self.query_one("#personas-export-json", Button)
        json_button.display = export_json_applies
        json_button.disabled = not export_enabled
        json_button.tooltip = export_tooltip
        png_button = self.query_one("#personas-export-png", Button)
        png_button.display = export_png_applies
        png_button.disabled = not (export_enabled and kind == "character")
        png_button.tooltip = export_tooltip
        self.query_one("#personas-delete", Button).disabled = not selected

    @on(ListView.Selected, "#personas-conversations-list")
    def _conversation_selected(self, event: ListView.Selected) -> None:
        event.stop()
        conversation_id = self._conversation_lookup.get(str(event.item.id or ""))
        if conversation_id is not None:
            self.post_message(ConversationRowSelected(conversation_id))
