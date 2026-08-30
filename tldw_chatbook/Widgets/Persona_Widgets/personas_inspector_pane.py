"""Selected-item inspector pane for the Personas workbench."""

from __future__ import annotations

import asyncio
import re

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import QueryError
from textual.widgets import Button, Checkbox, ListItem, ListView, Static

from ...Constants import PERSONAS_CONVERSATIONS_PAGE_SIZE
from ..Console.console_image_viewer_modal import ClickableAvatarBox

from .personas_messages import ActorPackExportRequested, PersonaBuddyActionRequested
from .personas_pane_messages import (
    ConversationRowSelected,
    OlderConversationsRequested,
)

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


class PersonasInspectorPane(VerticalScroll):
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

    PersonasInspectorPane .personas-conversations-tail {
        height: auto;
        min-height: 2;
    }

    PersonasInspectorPane .personas-conversations-tail Static {
        height: auto;
        min-height: 2;
        text-wrap: wrap;
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
        self._card_actions_visible = True
        self._is_unsaved = False
        self._selected_kind: str | None = None
        self._console_actions_enabled = False
        self._console_action_block_reason = "select an item"
        self._provider_block_reason: str | None = None
        self._conversation_lookup: dict[str, str] = {}
        self._conversation_tail: ListItem | None = None
        self._conversation_tail_actionable = False
        self._conversation_tail_loading = False
        self._conversation_render_attempt: object | None = None
        self._conversation_render_lock = asyncio.Lock()
        self._tts_export_available = False
        self._buddy_source: str | None = None
        self._buddy_persona_id: str | None = None
        self._buddy_revision: int | None = None
        self._buddy_active = False
        self._buddy_profile_current = False
        self._buddy_owner_source: str | None = None
        self._buddy_owner_persona_id: str | None = None
        self._buddy_enabled = False
        self._buddy_open = True
        self._actor_pack_source: str | None = None
        self._actor_pack_kind: str | None = None
        self._actor_pack_local_id: str | None = None
        self._actor_pack_revision: int | None = None
        self._actor_pack_eligible = False
        self._actor_pack_reason: str | None = None
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

    def set_card_actions_visible(self, visible: bool) -> None:
        """Retain whether card-level actions belong in the current center view.

        Args:
            visible: Whether the current center view owns card-level actions.
        """
        self._card_actions_visible = bool(visible)
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
        # Read-only policy-rules summary (workspace-assistant-defaults
        # Task 11): hidden until a persona is selected, same kind-gated
        # rendering idiom as the conversations section below.
        policy_summary = Static(
            "Tool policy: no rules", id="personas-policy-rules-summary", markup=False
        )
        policy_summary.display = False
        yield policy_summary
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
                "Export Actor Pack",
                id="personas-export-actor-pack",
                disabled=True,
                classes="console-action-subdued actor-pack-export-action",
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
        self._actor_pack_source = None
        self._actor_pack_kind = None
        self._actor_pack_local_id = None
        self._actor_pack_revision = None
        self._actor_pack_eligible = False
        self._actor_pack_reason = None
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
        self._actor_pack_source = None
        self._actor_pack_kind = None
        self._actor_pack_local_id = None
        self._actor_pack_revision = None
        self._actor_pack_eligible = False
        self._actor_pack_reason = None
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

    def set_buddy_status(
        self,
        *,
        source: str | None,
        persona_id: str | None,
        enabled: bool,
        open: bool,
    ) -> None:
        """Apply the screen-owned Buddy selection and visibility snapshot."""

        self._buddy_owner_source = source
        self._buddy_owner_persona_id = persona_id
        self._buddy_enabled = enabled is True
        self._buddy_open = open is True
        self._apply_action_state()

    def set_tts_export_available(self, available: bool) -> None:
        """Expose explicit inclusion only when the selected card has a profile."""

        self._tts_export_available = bool(available)
        if not self._tts_export_available:
            self.query_one("#personas-export-include-tts", Checkbox).value = False
        self._apply_action_state()

    def set_actor_pack_export_state(
        self,
        *,
        source: str,
        actor_kind: str,
        local_actor_id: str,
        actor_revision: int,
        eligible: bool,
        reason: str | None = None,
    ) -> None:
        """Apply the screen-owned Actor Pack export eligibility snapshot."""

        self._actor_pack_source = source
        self._actor_pack_kind = actor_kind
        self._actor_pack_local_id = local_actor_id
        self._actor_pack_revision = actor_revision
        self._actor_pack_eligible = eligible is True
        self._actor_pack_reason = reason
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

    def show_policy_rules(self, rules: list[dict] | tuple[dict, ...]) -> None:
        """Render the selected persona's policy rules as a read-only summary.

        Visibility is kind-gated in ``_apply_action_state`` (personas only);
        this method only owns the copy.
        """
        rules = tuple(rules or ())
        if not rules:
            summary = "Tool policy: no rules (default posture)"
        else:
            lines = [f"Tool policy: {len(rules)} rule(s)"]
            for rule in rules:
                verb = "allow" if rule.get("allowed", True) else "deny"
                extras: list[str] = []
                if rule.get("require_confirmation"):
                    extras.append("confirm")
                if rule.get("max_calls_per_turn") is not None:
                    extras.append(f"cap {rule['max_calls_per_turn']}")
                suffix = f" ({', '.join(extras)})" if extras else ""
                lines.append(
                    f"{rule.get('rule_kind')}: {rule.get('rule_name')} "
                    f"→ {verb}{suffix}"
                )
            summary = "\n".join(lines)
        try:
            self.query_one("#personas-policy-rules-summary", Static).update(summary)
        except QueryError:
            pass

    async def show_conversations_loading(
        self, render_attempt: object | None = None
    ) -> bool:
        """Show a loading placeholder while the listing worker runs.

        Args:
            render_attempt: Optional token that owns this render attempt.

        Returns:
            Whether the loading placeholder still owns the visible render.
        """
        token = self._claim_conversation_render(render_attempt)
        async with self._conversation_render_lock:
            return await self._show_conversations_placeholder(
                "Loading conversations...",
                actionable=False,
                disabled=True,
                render_attempt=token,
            )

    def invalidate_conversation_render(
        self, render_attempt: object | None = None
    ) -> None:
        """Invalidate the matching list render, or the current one.

        Args:
            render_attempt: Token to invalidate, or ``None`` for the current token.
        """
        if (
            render_attempt is None
            or self._conversation_render_attempt is render_attempt
        ):
            self._conversation_render_attempt = None

    def _claim_conversation_render(self, render_attempt: object | None) -> object:
        """Claim list rendering for an explicit or fresh opaque token."""
        token = render_attempt if render_attempt is not None else object()
        self._conversation_render_attempt = token
        return token

    def _standalone_or_existing_conversation_render(
        self, render_attempt: object | None
    ) -> object:
        """Use an explicit owner, or claim a fresh owner for legacy calls."""
        if render_attempt is not None:
            return render_attempt
        return self._claim_conversation_render(None)

    def _conversation_render_is_current(self, render_attempt: object) -> bool:
        return self._conversation_render_attempt is render_attempt

    async def _show_conversations_placeholder(
        self,
        text: str,
        *,
        actionable: bool = False,
        disabled: bool = True,
        render_attempt: object,
    ) -> bool:
        """Replace the rows with one conversation status or action line."""
        if not self._conversation_render_is_current(render_attempt):
            return False
        list_view = self.query_one("#personas-conversations-list", ListView)
        await list_view.clear()
        if not self._conversation_render_is_current(render_attempt):
            return False
        self._conversation_lookup = {}
        self._conversation_tail = None
        self._conversation_tail_actionable = False
        self._conversation_tail_loading = False
        tail = self._make_conversation_tail(
            text, actionable=actionable, disabled=disabled
        )
        await list_view.append(tail)
        return self._conversation_render_is_current(render_attempt)

    def _make_conversation_tail(
        self,
        text: str,
        *,
        actionable: bool,
        disabled: bool,
        loading: bool = False,
    ) -> ListItem:
        """Build and retain the exact trailing status/action row."""
        tail = ListItem(
            Static(text, markup=False),
            classes="personas-conversations-tail",
            disabled=disabled,
        )
        self._conversation_tail = tail
        self._conversation_tail_actionable = actionable
        self._conversation_tail_loading = loading
        return tail

    def _build_conversation_rows(
        self, rows: tuple[tuple[str, str], ...]
    ) -> list[ListItem]:
        """Build ordinary conversation rows through one durable lookup path."""
        items: list[ListItem] = []
        used_dom_ids = set(self._conversation_lookup)
        for conversation_id, title in rows:
            dom_id = (
                f"personas-conversation-row-{_ID_SAFE.sub('-', str(conversation_id))}"
            )
            if dom_id in used_dom_ids:
                suffix = 2
                while f"{dom_id}-{suffix}" in used_dom_ids:
                    suffix += 1
                dom_id = f"{dom_id}-{suffix}"
            used_dom_ids.add(dom_id)
            self._conversation_lookup[dom_id] = conversation_id
            items.append(
                ListItem(
                    Static(title, markup=False),
                    id=dom_id,
                    classes="personas-conversation-row console-action-subdued",
                )
            )
        return items

    async def _replace_conversation_tail(
        self,
        text: str,
        *,
        actionable: bool,
        disabled: bool,
        loading: bool = False,
        render_attempt: object,
    ) -> bool:
        """Replace only the current tail, retaining durable row widgets."""
        if not self._conversation_render_is_current(render_attempt):
            return False
        list_view = self.query_one("#personas-conversations-list", ListView)
        old_tail = self._conversation_tail
        if old_tail is not None and old_tail.is_mounted:
            old_tail.query_one(Static).update(text)
            old_tail.disabled = disabled
            self._conversation_tail_actionable = actionable
            self._conversation_tail_loading = loading
            old_tail.refresh(layout=True)
            return self._conversation_render_is_current(render_attempt)
        tail = self._make_conversation_tail(
            text,
            actionable=actionable,
            disabled=disabled,
            loading=loading,
        )
        await list_view.append(tail)
        return self._conversation_render_is_current(render_attempt)

    async def show_conversations(
        self,
        rows: tuple[tuple[str, str], ...],
        *,
        empty_copy: str | None = None,
        has_more: bool | None = None,
        render_attempt: object | None = None,
    ) -> bool:
        """Render (conversation_id, title) rows.

        An empty ``rows`` tuple clears the panel silently unless
        ``empty_copy`` is given, in which case that copy renders as a
        disabled placeholder (the library empty-state idiom). Supplying
        ``has_more`` opts into explicit empty/load/exhausted tail states.

        Args:
            rows: Conversation ID and display-title pairs.
            empty_copy: Optional copy for an empty result.
            has_more: Whether another saved-conversation page is available.
            render_attempt: Optional token that owns this render attempt.

        Returns:
            Whether this attempt still owns the visible render.
        """
        token = self._standalone_or_existing_conversation_render(render_attempt)
        async with self._conversation_render_lock:
            if not self._conversation_render_is_current(token):
                return False
            list_view = self.query_one("#personas-conversations-list", ListView)
            await list_view.clear()
            if not self._conversation_render_is_current(token):
                return False
            self._conversation_lookup = {}
            self._conversation_tail = None
            self._conversation_tail_actionable = False
            self._conversation_tail_loading = False
            if has_more is None and not rows and empty_copy:
                tail = self._make_conversation_tail(
                    empty_copy, actionable=False, disabled=True
                )
                await list_view.append(tail)
                return self._conversation_render_is_current(token)
            items = self._build_conversation_rows(rows)
            if has_more is not None:
                if not rows:
                    tail_copy = empty_copy or "No saved conversations."
                    actionable = False
                    disabled = True
                elif has_more:
                    tail_copy = (
                        f"Load {PERSONAS_CONVERSATIONS_PAGE_SIZE} "
                        "older conversations"
                    )
                    actionable = True
                    disabled = False
                else:
                    tail_copy = "All conversations shown."
                    actionable = False
                    disabled = False
                items.append(
                    self._make_conversation_tail(
                        tail_copy, actionable=actionable, disabled=disabled
                    )
                )
            if items:
                await list_view.extend(items)
            return self._conversation_render_is_current(token)

    async def show_older_conversations_loading(
        self, render_attempt: object | None = None
    ) -> bool:
        """Keep durable rows while replacing the action tail with busy copy.

        Args:
            render_attempt: Optional token that owns this render attempt.

        Returns:
            Whether this attempt still owns the visible render.
        """
        token = self._claim_conversation_render(render_attempt)
        async with self._conversation_render_lock:
            if not self._conversation_render_is_current(token):
                return False
            list_view = self.query_one("#personas-conversations-list", ListView)
            old_tail = self._conversation_tail
            clear_unfocused_tail = bool(
                old_tail is not None
                and not list_view.has_focus
                and list_view.highlighted_child is old_tail
            )
            rendered = await self._replace_conversation_tail(
                "Loading older conversations...",
                actionable=False,
                disabled=False,
                loading=True,
                render_attempt=token,
            )
            if rendered and clear_unfocused_tail:
                list_view.index = None
            return rendered

    async def show_conversations_failure(
        self,
        *,
        initial: bool,
        render_attempt: object | None = None,
        preserved_rows: tuple[tuple[str, str], ...] | None = None,
    ) -> bool:
        """Show an actionable initial or append retry state.

        Args:
            initial: Whether the failed request was the initial page.
            render_attempt: Optional token that owns this render attempt.
            preserved_rows: Durable rows to retain for an append retry.

        Returns:
            Whether this attempt still owns the visible render.
        """
        token = self._standalone_or_existing_conversation_render(render_attempt)
        async with self._conversation_render_lock:
            if initial:
                return await self._show_conversations_placeholder(
                    "Load failed.\nRetry conversations",
                    actionable=True,
                    disabled=False,
                    render_attempt=token,
                )
            if preserved_rows is not None:
                if not self._conversation_render_is_current(token):
                    return False
                list_view = self.query_one("#personas-conversations-list", ListView)
                # A result failure may follow a completed append. Clear finishes
                # before rebuilding the committed rows, so no candidate row is
                # retained while the retry tail is presented.
                await list_view.clear()
                if not self._conversation_render_is_current(token):
                    return False
                self._conversation_lookup = {}
                self._conversation_tail = None
                self._conversation_tail_actionable = False
                self._conversation_tail_loading = False
                items = self._build_conversation_rows(preserved_rows)
                items.append(
                    self._make_conversation_tail(
                        "Load failed.\nRetry older conversations",
                        actionable=True,
                        disabled=False,
                    )
                )
                await list_view.extend(items)
                return self._conversation_render_is_current(token)
            return await self._replace_conversation_tail(
                "Load failed.\nRetry older conversations",
                actionable=True,
                disabled=False,
                render_attempt=token,
            )

    async def append_conversations(
        self,
        rows: tuple[tuple[str, str], ...],
        *,
        has_more: bool,
        render_attempt: object | None = None,
    ) -> bool:
        """Append ordinary rows and replace only the pagination tail.

        Args:
            rows: Conversation ID and display-title pairs to append.
            has_more: Whether another saved-conversation page is available.
            render_attempt: Optional token that owns this render attempt.

        Returns:
            Whether this attempt still owns the visible render.
        """
        token = self._standalone_or_existing_conversation_render(render_attempt)
        async with self._conversation_render_lock:
            if not self._conversation_render_is_current(token):
                return False
            list_view = self.query_one("#personas-conversations-list", ListView)
            old_tail = self._conversation_tail
            highlighted_before = list_view.highlighted_child
            index_before = list_view.index
            may_advance_from_loading_tail = bool(
                self._conversation_tail_loading
                and old_tail is not None
                and list_view.has_focus
                and highlighted_before is old_tail
            )
            new_items = self._build_conversation_rows(rows)
            first_new_index = len(list_view.children) - (
                1 if old_tail is not None and old_tail.is_mounted else 0
            )
            if new_items:
                if old_tail is not None and old_tail.is_mounted:
                    await list_view.mount(*new_items, before=old_tail)
                else:
                    await list_view.extend(new_items)
                if not self._conversation_render_is_current(token):
                    return False
            rendered = await self._replace_conversation_tail(
                (
                    f"Load {PERSONAS_CONVERSATIONS_PAGE_SIZE} older conversations"
                    if has_more
                    else "All conversations shown."
                ),
                actionable=has_more,
                disabled=False,
                render_attempt=token,
            )
            if not rendered:
                return False
            if highlighted_before is old_tail and old_tail is not None:
                tail_index = list_view.children.index(old_tail)
                index_changed = list_view.index != index_before
                advance_from_loading_tail = bool(
                    may_advance_from_loading_tail
                    and new_items
                    and list_view.has_focus
                    and not index_changed
                )
                target_index = (
                    first_new_index
                    if advance_from_loading_tail
                    else list_view.index if index_changed else tail_index
                )
                list_view.index = tail_index
                if target_index != tail_index:
                    list_view.index = target_index
            return self._conversation_render_is_current(token)

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
        self.query_one("#personas-inspector-actions", Vertical).display = (
            selected and self._card_actions_visible
        )
        # Policy rules are a persona-record attribute (task-11); the section
        # hides for every other kind (the task-443 kind idiom).
        self.query_one(
            "#personas-policy-rules-summary", Static
        ).display = selected and kind == "persona"
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
        actor_pack_button = self.query_one("#personas-export-actor-pack", Button)
        actor_pack_applies = selected and kind in {"character", "persona"}
        actor_pack_enabled = bool(
            actor_pack_applies
            and not unsaved
            and marked == 0
            and self._actor_pack_eligible
            and self._actor_pack_source == "local"
            and self._actor_pack_kind == kind
            and self._actor_pack_local_id
            and type(self._actor_pack_revision) is int
            and self._actor_pack_revision >= 1
        )
        actor_pack_button.display = actor_pack_applies
        actor_pack_button.disabled = not actor_pack_enabled
        actor_pack_button.tooltip = (
            None
            if actor_pack_enabled
            else (
                _UNSAVED_TOOLTIP
                if unsaved
                else (
                    "Select one item to export."
                    if marked
                    else self._actor_pack_reason or _NO_SELECTION_EXPORT_TOOLTIP
                )
            )
        )
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
        use_button = self.query_one("#personas-buddy-use", Button)
        use_button.display = buddy_applies
        use_button.disabled = not buddy_eligible
        use_button.tooltip = None if buddy_eligible else buddy_tooltip
        owner = bool(
            buddy_eligible
            and self._buddy_owner_source == "local"
            and self._buddy_owner_persona_id == self._buddy_persona_id
        )
        owner_tooltip = "Select the Persona currently used by Buddy"
        if not buddy_eligible:
            state = {
                "personas-buddy-show": (False, buddy_tooltip),
                "personas-buddy-close": (False, buddy_tooltip),
                "personas-buddy-disable": (False, buddy_tooltip),
            }
        elif not owner:
            state = {
                "personas-buddy-show": (False, owner_tooltip),
                "personas-buddy-close": (False, owner_tooltip),
                "personas-buddy-disable": (False, owner_tooltip),
            }
        elif not self._buddy_enabled:
            disabled_tooltip = "Buddy is disabled. Use for Buddy to enable it."
            state = {
                "personas-buddy-show": (False, disabled_tooltip),
                "personas-buddy-close": (False, disabled_tooltip),
                "personas-buddy-disable": (False, "Buddy is already disabled."),
            }
        elif self._buddy_open:
            state = {
                "personas-buddy-show": (False, "Buddy is already open."),
                "personas-buddy-close": (True, None),
                "personas-buddy-disable": (True, None),
            }
        else:
            state = {
                "personas-buddy-show": (True, None),
                "personas-buddy-close": (False, "Buddy is already closed."),
                "personas-buddy-disable": (True, None),
            }
        for button_id, (enabled, tooltip) in state.items():
            button = self.query_one(f"#{button_id}", Button)
            button.display = buddy_applies
            button.disabled = not enabled
            button.tooltip = tooltip

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

    @on(Button.Pressed, "#personas-export-actor-pack")
    def _actor_pack_export_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if (
            event.button.disabled
            or self._actor_pack_kind not in {"character", "persona"}
            or self._actor_pack_source not in {"local", "server"}
            or not self._actor_pack_local_id
            or type(self._actor_pack_revision) is not int
        ):
            return
        self.post_message(
            ActorPackExportRequested(
                actor_kind=self._actor_pack_kind,
                source=self._actor_pack_source,
                local_actor_id=self._actor_pack_local_id,
                actor_revision=self._actor_pack_revision,
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
        elif (
            event.item is self._conversation_tail
            and self._conversation_tail_actionable
        ):
            self.post_message(OlderConversationsRequested())
