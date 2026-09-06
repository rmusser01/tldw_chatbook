"""Bounded Character conversation browser for the Console Context rail."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import Any, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches, QueryError
from textual.events import Click, DescendantBlur, DescendantFocus, Key
from textual.widgets import Button, Input, Static

from ...Character_Chat.character_conversation_navigation import (
    CharacterConversationGroup,
    CharacterConversationKey,
    CharacterConversationRow,
    ResolvedLocalCharacterKey,
    serialize_character_conversation_key,
)
from ...UI.Console_Modules.character_context import (
    ConsoleCharacterContextController,
    ConsoleCharacterContextState,
    ConsoleCharacterFocusIdentity,
    ConsoleCharacterOperationPhase,
    console_character_unavailable_reason_copy,
)
from ...Workspaces.conversation_browser_state import format_console_relative_age

CONSOLE_CHARACTER_CONTEXT_ID = "console-character-context"
CONSOLE_CHARACTER_SEARCH_ID = "console-character-search"


def _identity_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class CharacterConversationButton(Button):
    """Native row button whose single click selects and double click opens."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter,space", "press", "Open chat", show=False)
    ]

    def __init__(
        self, *args: Any, row: CharacterConversationRow, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.character_row = row

    async def _on_click(self, event: Click) -> None:
        event.prevent_default()
        event.stop()
        self.focus()
        if event.chain >= 2 and not self.disabled:
            self.press()


class CharacterGroupButton(Button):
    """Native accordion heading with conventional Enter and Space toggles."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter,space", "press", "Toggle group", show=False)
    ]


class ConsoleCharacterContext(Vertical):
    """Always-mounted, local-only Character browser without a nested scroller."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "clear_search", "Clear search", show=False)
    ]

    def __init__(
        self,
        controller: ConsoleCharacterContextController,
        *,
        identity_state: Static | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(id=CONSOLE_CHARACTER_CONTEXT_ID, **kwargs)
        self._controller = controller
        self._state = controller.state
        self._identity_state = identity_state
        self._task: asyncio.Task[Any] | None = None
        self._browse_focus: ConsoleCharacterFocusIdentity | None = None
        self._recompose_generation = 0
        self._focus_intent_generation = 0
        self._pending_focus_restore: ConsoleCharacterFocusIdentity | None = None

    @property
    def state(self) -> ConsoleCharacterContextState:
        return self._state

    @staticmethod
    def group_dom_id(key: CharacterConversationKey) -> str:
        """Return a stable DOM id derived from the complete typed key."""

        digest = _identity_digest(serialize_character_conversation_key(key))
        return f"console-character-group-{digest}"

    @staticmethod
    def row_dom_id(row_key: str) -> str:
        """Return a stable DOM id derived from the repository row identity."""

        return f"console-character-row-{_identity_digest({'row_key': row_key})}"

    @staticmethod
    def action_dom_id(role: str, key: CharacterConversationKey) -> str:
        """Return a stable DOM id for a group-scoped action."""

        digest = _identity_digest(serialize_character_conversation_key(key))
        return f"console-character-{role}-{digest}"

    def on_mount(self) -> None:
        self.watch(self.screen, "focused", self._observe_focus_intent)
        self._start(self._controller.refresh_if_scope_changed())

    def _observe_focus_intent(self, old: Any, new: Any) -> None:
        """External focus wins; pruning the owning node is not a new intent."""
        if new is not None and new is not self and self not in new.ancestors:
            self._focus_intent_generation += 1
            self._pending_focus_restore = None

    def _start(self, coroutine: Any) -> None:
        self._task = asyncio.create_task(coroutine)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "clear_search":
            return bool(
                self._state.query
                or self._state.phase is ConsoleCharacterOperationPhase.OPENING
            )
        return True

    def action_clear_search(self) -> None:
        """Clear search or cancel an activation before its commit point."""

        if self._state.phase is ConsoleCharacterOperationPhase.OPENING:
            self._controller.cancel_activation()
            return
        if self._state.query:
            try:
                self.query_one(f"#{CONSOLE_CHARACTER_SEARCH_ID}", Input).value = ""
            except (NoMatches, QueryError):
                self._start(self._controller.search(""))

    @staticmethod
    def _focus_identity(widget: object) -> ConsoleCharacterFocusIdentity | None:
        identity = getattr(widget, "character_focus_identity", None)
        return identity if isinstance(identity, ConsoleCharacterFocusIdentity) else None

    def sync_state(self, state: ConsoleCharacterContextState) -> None:
        """Apply one complete snapshot and semantically restore focus/scroll."""

        if not self.is_attached or not self.app.screen_stack or state == self._state:
            return
        current_focus = (
            self._focus_identity(self.app.focused) if self.is_mounted else None
        )
        self._state = state
        self._sync_identity_line()
        restore = state.restore_focus or current_focus
        if restore is None and self.app.focused is None:
            restore = self._pending_focus_restore
        self._pending_focus_restore = restore
        self._recompose_generation += 1
        generation = self._recompose_generation
        asyncio.create_task(
            self._recompose_and_restore(
                generation,
                restore,
                state.restore_scroll_offset,
                self._focus_intent_generation,
            )
        )

    async def _recompose_and_restore(
        self,
        generation: int,
        focus: ConsoleCharacterFocusIdentity | None,
        scroll_offset: int | None,
        focus_intent_generation: int,
    ) -> None:
        """Restore only after Textual has mounted the new semantic projection."""

        if not self.is_attached or not self.app.screen_stack:
            return
        focused = self.app.focused
        if focused is not None and self in focused.ancestors:
            # Avoid Textual's automatic external fallback while pruning the
            # old node. A real subsequent external focus still advances intent.
            self.screen.set_focus(None)
        await self.recompose()
        if (
            self.is_attached
            and bool(self.app.screen_stack)
            and generation == self._recompose_generation
            and focus_intent_generation == self._focus_intent_generation
        ):
            self._restore_browse_position(focus, scroll_offset)
            self._pending_focus_restore = None

    def _outer_scroll(self) -> Any | None:
        try:
            return self.screen.query_one("#console-left-rail-body")
        except (NoMatches, QueryError):
            return None

    def _resolve_focus_id(self, identity: ConsoleCharacterFocusIdentity | None) -> str:
        groups = self._state.groups
        keys = {group.key for group in groups}
        if identity is not None:
            if identity.role == "search":
                return CONSOLE_CHARACTER_SEARCH_ID
            if identity.role == "row" and identity.row_key:
                rows = (
                    *self._state.search_rows,
                    *(row for group in groups for row in group.rows),
                )
                if any(row.row_key == identity.row_key for row in rows):
                    return self.row_dom_id(identity.row_key)
            if identity.group_key in keys:
                return self.group_dom_id(identity.group_key)
        fallback = self._state.expanded_key
        if fallback not in keys:
            fallback = next((group.key for group in groups if group.is_current), None)
        if fallback is None and groups:
            fallback = groups[0].key
        if fallback is not None:
            return self.group_dom_id(fallback)
        return CONSOLE_CHARACTER_SEARCH_ID

    def _restore_browse_position(
        self,
        focus: ConsoleCharacterFocusIdentity | None,
        scroll_offset: int | None,
    ) -> None:
        if focus is not None:
            try:
                self.query_one(f"#{self._resolve_focus_id(focus)}").focus()
            except (NoMatches, QueryError):
                pass
        outer = self._outer_scroll()
        if outer is not None and scroll_offset is not None:
            outer.scroll_to(y=scroll_offset, animate=False, immediate=True, force=True)

    def _restore_focus(self, identity: ConsoleCharacterFocusIdentity) -> None:
        self._restore_browse_position(identity, None)

    @staticmethod
    def _row_label(row: CharacterConversationRow) -> str:
        suffix = " · current" if row.is_current else ""
        return f"{row.title or 'Untitled chat'}{suffix}"

    @staticmethod
    def _search_row_label(row: CharacterConversationRow) -> str:
        age = format_console_relative_age(row.last_modified, now=datetime.now(UTC))
        suffix = f" · {age}" if age else ""
        return f"{row.title or 'Untitled chat'}\n{row.character_label}\nLocal{suffix}"

    @staticmethod
    def _group_label(group: CharacterConversationGroup, expanded: bool) -> str:
        glyph = "▾" if expanded else "▸"
        current = " · current" if group.is_current else ""
        return f"{glyph} {group.character_label} · {group.total} chats{current}"

    def _reason_for(self, row: CharacterConversationRow) -> str:
        detail = self._state.unavailable_detail(row.row_key)
        return (
            detail.reason_copy
            if detail is not None
            else console_character_unavailable_reason_copy(row.unavailable_reason)
        )

    def _sync_identity_line(self) -> None:
        if self._identity_state is None:
            return
        scope = self._state.scope_fingerprint
        has_current = scope is not None and scope.current_character_id is not None
        if not has_current:
            copy = "No current character · Local · No open chat"
        else:
            character = scope.current_character_label or "Current character"
            open_copy = "Open" if scope.open_conversation_id else "No open chat"
            copy = f"{character} · Local · {open_copy}"
        self._identity_state.update(copy)
        self._identity_state.tooltip = copy

    def _compose_unavailable_detail(
        self, row: CharacterConversationRow
    ) -> ComposeResult:
        detail = self._state.unavailable_detail(row.row_key)
        reason = detail.reason_copy if detail is not None else self._reason_for(row)
        yield Static(reason, id="console-character-unavailable-reason", markup=False)
        opening = (
            self._state.phase is ConsoleCharacterOperationPhase.REPAIRING
            and self._state.operation_row_key == row.row_key
        )
        open_button = Button(
            "Opening Library…" if opening else "Open in Library",
            id="console-character-open-library",
            classes="console-character-action",
            compact=True,
            disabled=opening,
        )
        open_button.character_row = row
        open_button.character_unavailable_action = "open"
        yield open_button
        if detail is not None and detail.can_repair:
            repair = Button(
                "Opening repair…" if opening else "Repair in Library",
                id="console-character-repair-library",
                classes="console-character-action",
                compact=True,
                disabled=opening,
            )
            repair.character_row = row
            repair.character_unavailable_action = "repair"
            yield repair

    def _compose_group(self, group: CharacterConversationGroup) -> ComposeResult:
        expanded = self._state.expanded_key == group.key
        identity = ConsoleCharacterFocusIdentity("group", group_key=group.key)
        header = CharacterGroupButton(
            self._group_label(group, expanded),
            id=self.group_dom_id(group.key),
            classes="console-character-group",
            compact=True,
        )
        header.character_group = group
        header.character_focus_identity = identity
        header.tooltip = self._group_label(group, expanded).lstrip("▾▸ ")
        yield header
        if not expanded:
            return
        if not group.rows:
            yield Static(
                f"No chats with {group.character_label} yet",
                classes="console-character-empty",
                markup=False,
            )
            if group.is_current and isinstance(group.key, ResolvedLocalCharacterKey):
                start = Button(
                    "Start in Console",
                    id=self.action_dom_id("start", group.key),
                    classes="console-character-action",
                    compact=True,
                )
                start.character_group = group
                yield start
        for row in group.rows[:5]:
            selected = self._state.selected_unavailable_row_key == row.row_key
            label = self._row_label(row)
            if row.unresolved is not None:
                label = f"{label} · {self._reason_for(row)}"
            opening = (
                self._state.phase is ConsoleCharacterOperationPhase.OPENING
                and self._state.operation_row_key == row.row_key
            )
            if opening:
                label = f"{label} · Opening…"
            button = CharacterConversationButton(
                label,
                row=row,
                id=self.row_dom_id(row.row_key),
                classes=(
                    "console-character-row -opening"
                    if opening
                    else "console-character-row"
                ),
                compact=True,
            )
            button.character_focus_identity = ConsoleCharacterFocusIdentity(
                "row", group_key=group.key, row_key=row.row_key
            )
            button.tooltip = label
            yield button
            if row.unresolved is not None and selected:
                yield from self._compose_unavailable_detail(row)
        if group.total:
            action_label = (
                f"View all {group.total} in Roleplay"
                if isinstance(group.key, ResolvedLocalCharacterKey)
                else f"View all {group.total} in Library"
            )
            action = Button(
                action_label,
                id=self.action_dom_id("view-all", group.key),
                classes="console-character-action",
                compact=True,
            )
            action.character_group = group
            action.tooltip = action_label
            yield action

    def _compose_search_rows(self) -> ComposeResult:
        if not self._state.search_rows:
            yield Static(
                "No local character chats match",
                classes="console-character-empty",
                markup=False,
            )
        for row in self._state.search_rows[:8]:
            label = self._search_row_label(row)
            if row.unresolved is not None:
                label = f"{label} · {self._reason_for(row)}"
            opening = (
                self._state.phase is ConsoleCharacterOperationPhase.OPENING
                and self._state.operation_row_key == row.row_key
            )
            if opening:
                label = f"{label} · Opening…"
            button = CharacterConversationButton(
                label,
                row=row,
                id=self.row_dom_id(row.row_key),
                classes=(
                    "console-character-row console-character-search-row -opening"
                    if opening
                    else "console-character-row console-character-search-row"
                ),
                compact=True,
            )
            button.character_focus_identity = ConsoleCharacterFocusIdentity(
                "row", row_key=row.row_key
            )
            button.tooltip = label
            yield button

    def compose(self) -> ComposeResult:
        self._sync_identity_line()

        search = Input(
            value=self._state.query,
            placeholder="Search chats",
            name="Global Keyword search over local character chats",
            id=CONSOLE_CHARACTER_SEARCH_ID,
        )
        search.character_focus_identity = ConsoleCharacterFocusIdentity("search")
        search.tooltip = "Global Keyword search over local character chats"
        yield search

        if self._state.phase is ConsoleCharacterOperationPhase.REFRESHING:
            yield Static(
                "Refreshing local character chats…",
                id="console-character-loading",
                markup=False,
            )
            return
        if self._state.phase is ConsoleCharacterOperationPhase.SEARCHING:
            yield Static(
                "Searching local character chats…",
                id="console-character-loading",
                markup=False,
            )
            return
        if self._state.error:
            yield Static(self._state.error, id="console-character-error", markup=False)
            yield Button("Retry", id="console-character-retry", compact=True)
            return
        if self._state.query:
            yield from self._compose_search_rows()
            if self._controller.query_handoff_available:
                yield Button(
                    "Continue search in Character chats",
                    id="console-character-query-handoff",
                    compact=True,
                )
            return
        if not self._state.groups:
            yield Static(
                "No character chats yet",
                classes="console-character-empty",
                markup=False,
            )
            yield Button(
                "Open Roleplay", id="console-character-open-roleplay", compact=True
            )
            return
        for group in self._state.groups[:4]:
            yield from self._compose_group(group)

    @on(Input.Changed, f"#{CONSOLE_CHARACTER_SEARCH_ID}")
    def _search_changed(self, event: Input.Changed) -> None:
        from ...Utils.input_validation import (
            CONSOLE_SWITCHER_QUERY_MAX_LENGTH,
            validate_console_switcher_query,
        )

        try:
            query = validate_console_switcher_query(event.value).strip()
        except ValueError:
            event.input.value = self._state.query
            self.notify(
                f"Search must be text, at most {CONSOLE_SWITCHER_QUERY_MAX_LENGTH} "
                "characters. Previous search kept.",
                severity="warning",
            )
            return
        if query == self._state.query:
            return
        if query and not self._state.query:
            outer = self._outer_scroll()
            focused = self.app.focused
            self._controller.capture_browse(
                focus=self._browse_focus or self._focus_identity(focused),
                scroll_offset=int(getattr(outer, "scroll_y", 0) or 0),
            )
        self._start(self._controller.search(query))

    def on_descendant_blur(self, event: DescendantBlur) -> None:
        """Remember the last stable browse target before focus enters search."""

        identity = self._focus_identity(event.widget)
        if not self._state.query and identity is not None and identity.role != "search":
            self._browse_focus = identity

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Select unavailable detail without treating focus as activation."""

        row = getattr(event.widget, "character_row", None)
        if (
            isinstance(row, CharacterConversationRow)
            and row.unresolved is not None
            and self._state.selected_unavailable_row_key != row.row_key
        ):
            self._controller.select_unavailable(row.row_key)
            self.sync_state(self._controller.state)

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        button = event.button
        group = getattr(button, "character_group", None)
        if button.id == "console-character-retry":
            self._start(
                self._controller.search(self._state.query)
                if self._state.query
                else self._controller.refresh()
            )
            return
        if button.id == "console-character-open-roleplay":
            self._controller.open_roleplay_home()
            return
        if button.id == "console-character-query-handoff":
            self._controller.handoff_query(self._state.query)
            return
        if isinstance(button, CharacterGroupButton) and group is not None:
            identity = ConsoleCharacterFocusIdentity("group", group_key=group.key)
            self._controller.toggle_group(group.key)
            self.sync_state(self._controller.state)
            self.call_after_refresh(self._restore_focus, identity)
            return
        if button.id and button.id.startswith("console-character-start-") and group:
            self._start(self._controller.start_current(group))
            return
        if button.id and button.id.startswith("console-character-view-all-") and group:
            self._start(self._controller.view_group(group))
            return
        unavailable_action = getattr(button, "character_unavailable_action", "")
        row = getattr(button, "character_row", None)
        if unavailable_action and isinstance(row, CharacterConversationRow):
            if row.unresolved is None:
                return
            route = (
                self._controller.repair_unavailable
                if unavailable_action == "repair"
                else self._controller.open_unavailable
            )
            self._start(route(row.unresolved, row_key=row.row_key))
            return
        if isinstance(button, CharacterConversationButton):
            if self._state.phase is ConsoleCharacterOperationPhase.OPENING:
                return
            row = button.character_row
            if row.target is not None:
                self._start(self._controller.activate(row.target, row_key=row.row_key))
            elif row.unresolved is not None:
                self._start(
                    self._controller.open_unavailable(
                        row.unresolved, row_key=row.row_key
                    )
                )

    def on_key(self, event: Key) -> None:
        focused = self.app.focused
        group = getattr(focused, "character_group", None)
        if group is not None and event.key in {"left", "right"}:
            expanded = self._state.expanded_key == group.key
            if (event.key == "left" and expanded) or (
                event.key == "right" and not expanded
            ):
                event.stop()
                identity = self._focus_identity(focused)
                self._controller.toggle_group(group.key)
                self.sync_state(self._controller.state)
                if identity is not None:
                    self.call_after_refresh(self._restore_focus, identity)
            return
        if event.key == "escape":
            if self._state.phase is ConsoleCharacterOperationPhase.OPENING:
                event.stop()
                self._controller.cancel_activation()
                return
            if self._state.query:
                event.stop()
                try:
                    self.query_one(f"#{CONSOLE_CHARACTER_SEARCH_ID}", Input).value = ""
                except (NoMatches, QueryError):
                    self._start(self._controller.search(""))
