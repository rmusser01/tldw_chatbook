"""Console-native run inspector."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches, QueryError
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import (
    ConsoleInspectorAction,
    ConsoleInspectorState,
    normalize_console_source_status,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard
from tldw_chatbook.Widgets.Console.console_inspector_ownership import (
    ACTION_GROUPS,
    ROW_GROUPS as _ROW_GROUPS,
    InspectorOwnedContent,
    InspectorOwnershipPolicy,
    classify_inspector_content,
)
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)

_ACTION_GROUPS = ACTION_GROUPS
_CONDITIONAL_OWNERS = ("Tools", "Approvals", "Artifacts")
_DUPLICATE_PINNED_LABELS = {"Selected conversation", "Workspace"}


def _row_status_class(status: str) -> str:
    """Return the one status class an Inspector row may carry.

    TASK-24608: this used to be ``f"console-inspector-row-{status}"`` with the
    producer's raw string, which had two consequences. The set of possible
    selectors was open -- any status a producer invented became a class
    nothing could style -- and, more simply, *no* stylesheet in the repo
    defined any member of the family, so the whole channel painted nothing
    while being swapped on every in-place update and asserted by tests.
    Normalizing closes the set to the four ``normalize_console_source_status``
    guarantees, which are exactly the four the stylesheet now defines, so a
    class can no longer exist without a rule behind it.

    Args:
        status: Raw status string from a display row.

    Returns:
        One of the four ``console-inspector-row-*`` class names.
    """

    return f"console-inspector-row-{normalize_console_source_status(status)}"


def inspector_group_is_actionable(owner: str, owned: InspectorOwnedContent) -> bool:
    """Return owner-specific actionability for a conditional Inspector group.

    Args:
        owner: The conditional group name.
        owned: The classified Inspector content.

    Returns:
        True when the group should be promoted above More.

    Raises:
        ValueError: If ``owner`` is not a conditional group.
    """

    if owner not in _CONDITIONAL_OWNERS:
        raise ValueError(owner)
    actions = owned.actions_for(owner)
    if any(action.enabled for action in actions):
        return True
    rows = owned.rows_for(owner)
    values = {str(entry.row.value).strip().lower() for entry in rows}
    if owner == "Tools":
        return any(value not in {"", "—", "0", "0 ready"} for value in values)
    if owner == "Approvals":
        return any(entry.row.status == "blocked" for entry in rows)
    return any(
        value not in {"", "—", "none", "unavailable", "not available for this item"}
        for value in values
    )


class ConsoleInspectorMoreButton(Button):
    """Native Button press path with terminal Space parity."""

    def action_press(self) -> None:
        """Allow consecutive keyboard toggles during Button's paint effect."""

        self.press()

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "space":
            event.stop()
            event.prevent_default()
            self.press()
            return
        await super()._on_key(event)


class ConsoleInspectorMore(Vertical):
    """Disclosure boundary for non-actionable conditional groups."""

    BINDINGS = [
        Binding("left", "collapse", "Collapse", show=False),
        Binding("right", "expand", "Expand", show=False),
    ]

    class Toggled(Message):
        def __init__(self, open: bool) -> None:
            super().__init__()
            self.open = open

    def __init__(self, *children: Widget, open: bool) -> None:
        self.open = open
        toggle = ConsoleInspectorMoreButton(
            "More", id="console-inspector-more-toggle", compact=True
        )
        body = Vertical(*children, id="console-inspector-more-body")
        body.display = open
        body.styles.display = "block" if open else "none"
        super().__init__(toggle, body, id="console-inspector-more")
        self.styles.height = "auto" if open else 2

    def on_mount(self) -> None:
        """Apply the initial disclosure state after mounting."""

        # Parent Mount may be delivered before composed children have
        # completed their own mount under a heavily loaded message pump.
        self.call_after_refresh(self._apply_open)

    def _apply_open(self) -> None:
        # A deferred initial-state callback may outlive this disclosure when
        # the user navigates away during a busy frame. Its children have
        # already unmounted in that case, so there is nothing left to apply.
        bodies = self.query("#console-inspector-more-body")
        if not self.is_mounted or not bodies:
            return
        body = bodies.first(Vertical)
        body.display = self.open
        body.styles.display = "block" if self.open else "none"
        self.styles.height = "auto" if self.open else 2
        for heading in body.query(".console-inspector-group-heading"):
            heading.can_focus = self.open

    def set_open(self, open: bool, *, notify: bool = False) -> None:
        """Apply a disclosure state and optionally announce a user change.

        Args:
            open: Whether the conditional groups should be visible.
            notify: Whether to post a ``Toggled`` message.
        """

        if open == self.open:
            return
        focused = self.app.focused if self.is_mounted else None
        ancestor = focused
        focus_was_inside = False
        while isinstance(ancestor, Widget):
            if ancestor is self:
                focus_was_inside = True
                break
            ancestor = ancestor.parent
        self.open = open
        if self.is_mounted:
            self._apply_open()
            if not open and focus_was_inside:
                self.query_one("#console-inspector-more-toggle", Button).focus()
        if notify:
            self.post_message(self.Toggled(open))

    @on(Button.Pressed, "#console-inspector-more-toggle")
    def _toggle(self, event: Button.Pressed) -> None:
        event.stop()
        self.set_open(not self.open, notify=True)

    def action_collapse(self) -> None:
        """Collapse More and announce the deliberate action."""

        self.set_open(False, notify=True)

    def action_expand(self) -> None:
        """Expand More and announce the deliberate action."""

        self.set_open(True, notify=True)


class ConsoleRunInspector(RecomposeCaptureGuard, Vertical):
    """Render Console run readiness, recovery, and action affordances."""

    class MoreToggled(Message):
        """A deliberate user change to the local More disclosure."""

        def __init__(self, open: bool) -> None:
            super().__init__()
            self.open = open

    def __init__(
        self,
        state: ConsoleInspectorState,
        *,
        ownership_policy: InspectorOwnershipPolicy = InspectorOwnershipPolicy.STRICT,
        reported_unknown_fingerprints: set[tuple[str, ...]] | None = None,
        on_reconcile: Callable[[], None] | None = None,
        on_more_focus_removed: Callable[[str | None], None] | None = None,
        more_open: bool = False,
        **kwargs: Any,
    ) -> None:
        ownership = classify_inspector_content(state, ownership_policy)
        super().__init__(**kwargs)
        self.state = state
        self.ownership_policy = ownership_policy
        self._ownership = ownership
        self._reported_unknown_fingerprints = (
            reported_unknown_fingerprints
            if reported_unknown_fingerprints is not None
            else set()
        )
        self._on_reconcile = on_reconcile
        self._on_more_focus_removed = on_more_focus_removed
        self._more_open = more_open
        self._pending_conditional_focus: tuple[str, str | None] | None = None
        self._pending_more_focus_recovery = False
        self._pending_more_focus_section_id: str | None = None
        self._report_unowned_content(ownership)
        self.styles.height = "auto"
        self.styles.min_height = 0
        #: Count of wholesale recomposes taken by ``sync_state`` (test seam).
        self.recompose_count = 0

    def sync_state(self, state: ConsoleInspectorState) -> None:
        """Refresh the mounted inspector from a new display-state snapshot.

        TASK-259: when only row text/status changed (same rendered row ids,
        same actions, same dictionary section shape), the mounted row
        ``Static`` widgets are updated in place instead of tearing down and
        recomposing the whole inspector. Any structural change (rows added/
        removed/reordered, action or dictionary changes) still recomposes.

        Args:
            state: New inspector display-state snapshot.
        """
        if state == self.state:
            return
        previous = self.state
        previous_ownership = self._ownership
        ownership = classify_inspector_content(state, self.ownership_policy)
        self.state = state
        self._ownership = ownership
        self._report_unowned_content(ownership)
        structural_change = self._structural_key(
            previous, previous_ownership
        ) != self._structural_key(state, ownership)
        _previous_promoted, previous_more = self._group_projection(previous_ownership)
        _promoted, more = self._group_projection(ownership)
        focused = self.app.focused if self.is_mounted else None
        recover_removed_more_focus = False
        more_focus_section_id = None
        if (
            structural_change
            and previous_more
            and not more
            and isinstance(focused, Widget)
            and focused.id == "console-inspector-more-toggle"
            and self._on_more_focus_removed is not None
        ):
            recover_removed_more_focus = True
            more_focus_section_id = self._next_ordinary_section_after_more(ownership)
        focus_snapshot = (
            self._conditional_focus_snapshot(previous_ownership)
            if structural_change
            else None
        )
        if (
            not self.is_mounted
            or structural_change
            or not self._apply_row_updates(previous, previous_ownership)
        ):
            self._pending_conditional_focus = focus_snapshot
            self._pending_more_focus_recovery = recover_removed_more_focus
            self._pending_more_focus_section_id = more_focus_section_id
            self.recompose_count += 1
            self.refresh(recompose=True)
            return
        # Deferred to match the recompose path's timing: a wholesale
        # recompose lands on the NEXT refresh cycle, i.e. after any rail
        # cascade the owning screen applies later in the same sync tick.
        self.call_after_refresh(self._restore_rail_cascade_visibility)
        self.call_after_refresh(self._request_sections_reconcile)

    async def recompose(self) -> None:
        """Reconcile the replacement sections after their DOM is committed."""

        await super().recompose()
        self._request_sections_reconcile()
        focus_snapshot = self._pending_conditional_focus
        recover_removed_more_focus = self._pending_more_focus_recovery
        more_focus_section_id = self._pending_more_focus_section_id
        self._pending_conditional_focus = None
        self._pending_more_focus_recovery = False
        self._pending_more_focus_section_id = None
        if recover_removed_more_focus and self._on_more_focus_removed is not None:
            self.call_after_refresh(
                self._on_more_focus_removed,
                more_focus_section_id,
            )
        if focus_snapshot is not None:
            self.call_after_refresh(self._recover_conditional_focus, *focus_snapshot)

    def _report_unowned_content(self, ownership: InspectorOwnedContent) -> None:
        """Log one privacy-safe diagnostic for each unknown fingerprint."""
        fingerprint = ownership.unknown_identifiers
        if not fingerprint or fingerprint in self._reported_unknown_fingerprints:
            return
        self._reported_unknown_fingerprints.add(fingerprint)
        logger.warning("Inspector ownership incomplete: {}", fingerprint)

    def _restore_rail_cascade_visibility(self) -> None:
        """Mirror recompose semantics for the Console rail-collapse cascade.

        A wholesale recompose replaces every child, implicitly dropping the
        forced ``display=False`` (and its ``_console_rail_prior_display``
        marker) that ``ChatScreen._sync_console_rail_descendant_visibility``
        stamps on descendants while the inspector rail is collapsed. The
        in-place update path keeps the original children, so it must restore
        the same state explicitly or rows updated while the rail is hidden
        would stay ``display=False`` after the rail reopens mid-recompose
        (and diverge from the recompose path's observable DOM).
        """
        for child in self.query("*"):
            prior_display = getattr(child, "_console_rail_prior_display", None)
            if prior_display is None:
                continue
            child.display = bool(prior_display)
            child.styles.display = "block" if prior_display else "none"
            delattr(child, "_console_rail_prior_display")

    @classmethod
    def _group_projection(
        cls, ownership: InspectorOwnedContent
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        present = tuple(
            owner
            for owner in _CONDITIONAL_OWNERS
            if ownership.rows_for(owner) or ownership.actions_for(owner)
        )
        promoted = tuple(
            owner
            for owner in present
            if inspector_group_is_actionable(owner, ownership)
        )
        return promoted, tuple(owner for owner in present if owner not in promoted)

    @staticmethod
    def _rows_for(ownership: InspectorOwnedContent, owner: str) -> tuple:
        rows = ownership.rows_for(owner)
        if owner == "Selected Conversation":
            rows = tuple(
                entry
                for entry in rows
                if entry.row.label not in _DUPLICATE_PINNED_LABELS
            )
        return rows

    @classmethod
    def _rendered_row_entries(
        cls,
        state: ConsoleInspectorState,
        ownership: InspectorOwnedContent | None = None,
    ) -> list[tuple[str, str, str]]:
        """Return ``(widget_id, text, status)`` for each row ``compose`` mounts.

        Reads the same filtered projection as ``compose`` so per-row updates
        target exactly the mounted widgets.

        Args:
            state: Inspector display-state snapshot to project.

        Returns:
            Row entries in compose order, dictionary rows last.
        """
        owned = ownership or classify_inspector_content(
            state, InspectorOwnershipPolicy.STRICT
        )
        promoted, more = cls._group_projection(owned)
        ordinary = tuple(
            owner
            for owner, _heading_id, _labels in _ROW_GROUPS
            if owner not in _CONDITIONAL_OWNERS
        )
        ordered_owners = ordinary[:2] + promoted + more + ordinary[2:]
        projected_rows = (
            *(
                entry
                for owner in ordered_owners
                for entry in cls._rows_for(owned, owner)
            ),
            *owned.dictionary_rows,
            *owned.world_book_rows,
        )
        return [
            (entry.widget_id, entry.row.text, entry.row.status)
            for entry in projected_rows
        ]

    @classmethod
    def _structural_key(
        cls,
        state: ConsoleInspectorState,
        ownership: InspectorOwnedContent | None = None,
    ) -> tuple:
        """Return a key identifying the mounted widget structure for a state.

        Two states with equal keys mount the same widget ids in the same
        order with identical action buttons, so they differ at most in row
        text/status -- safe for in-place updates.

        Args:
            state: Inspector display-state snapshot to fingerprint.

        Returns:
            Hashable structure key (row ids + action tuples).
        """

        def _action_key(action: ConsoleInspectorAction) -> tuple:
            return (
                action.widget_id,
                action.label,
                action.enabled,
                getattr(action, "disabled_reason", ""),
                getattr(action, "tooltip", ""),
                getattr(action, "classes", ""),
            )

        owned = ownership or classify_inspector_content(
            state, InspectorOwnershipPolicy.STRICT
        )
        promoted, more = cls._group_projection(owned)
        return (
            tuple(entry[0] for entry in cls._rendered_row_entries(state, owned)),
            promoted,
            more,
            bool(more),
            tuple(_action_key(action) for action in owned.known_actions),
            tuple(_action_key(entry.action) for entry in owned.dictionary_actions),
            tuple(_action_key(entry.action) for entry in owned.world_book_actions),
        )

    def _conditional_focus_snapshot(
        self, ownership: InspectorOwnedContent
    ) -> tuple[str, str | None] | None:
        focused = self.app.focused if self.is_mounted else None
        if not isinstance(focused, Widget):
            return None
        focused_id = focused.id
        for owner in _CONDITIONAL_OWNERS:
            ids = {
                *(entry.widget_id for entry in ownership.rows_for(owner)),
                *(action.widget_id for action in ownership.actions_for(owner)),
                f"console-inspector-{owner.lower()}-heading",
            }
            if focused_id in ids:
                return owner, focused_id
        return None

    @staticmethod
    def _focusable(widget: Widget) -> bool:
        return bool(
            widget.can_focus
            and not widget.disabled
            and widget.display
            and widget.styles.display != "none"
        )

    def _recover_conditional_focus(self, owner: str, focused_id: str | None) -> None:
        if not self.is_mounted:
            return
        current = self.app.focused
        if (
            isinstance(current, Widget)
            and current.is_mounted
            and self not in current.ancestors
            and self._focusable(current)
        ):
            return
        promoted, more = self._group_projection(self._ownership)
        if focused_id and owner in promoted:
            try:
                same = self.query_one(f"#{focused_id}", Widget)
            except (NoMatches, QueryError):
                same = None
            if same is not None and self._focusable(same):
                same.focus()
                return
        if self._more_open and owner in more:
            if focused_id:
                try:
                    same = self.query_one(f"#{focused_id}", Widget)
                except (NoMatches, QueryError):
                    same = None
                if same is not None and self._focusable(same):
                    same.focus()
                    return
            try:
                heading = self.query_one(
                    f"#console-inspector-{owner.lower()}-heading", Widget
                )
            except (NoMatches, QueryError):
                heading = None
            if heading is not None and self._focusable(heading):
                heading.focus()
                return
        try:
            self.query_one("#console-inspector-more-toggle", Button).focus()
        except (NoMatches, QueryError):
            return

    def set_more_open(self, open: bool) -> None:
        """Apply persisted disclosure state without posting a user event.

        Args:
            open: Whether the conditional More groups should be visible.
        """

        self._more_open = open
        if not self.is_mounted:
            return
        try:
            more = self.query_one("#console-inspector-more", ConsoleInspectorMore)
        except (NoMatches, QueryError):
            return
        more.set_open(open)

    @on(ConsoleInspectorMore.Toggled)
    def _more_toggled(self, event: ConsoleInspectorMore.Toggled) -> None:
        event.stop()
        self._more_open = event.open
        self.post_message(self.MoreToggled(event.open))

    def _apply_row_updates(
        self,
        previous: ConsoleInspectorState,
        previous_ownership: InspectorOwnedContent,
    ) -> bool:
        """Update changed row Statics in place after a non-structural change.

        Args:
            previous: The state snapshot that produced the mounted rows.

        Returns:
            True when all changed rows were updated in place; False when a
            target widget was missing (caller falls back to recompose).
        """
        old_entries = self._rendered_row_entries(previous, previous_ownership)
        for (widget_id, text, status), (_old_id, old_text, old_status) in zip(
            self._rendered_row_entries(self.state, self._ownership), old_entries
        ):
            if text == old_text and status == old_status:
                continue
            try:
                row_widget = self.query_one(f"#{widget_id}", Static)
            except (NoMatches, QueryError):
                return False
            row_widget.update(text)
            if status != old_status:
                # Both sides normalize, so a swap between two synonyms of one
                # class (e.g. "missing" -> "unavailable") removes and re-adds
                # the same class rather than stranding a dead one.
                row_widget.remove_class(_row_status_class(old_status))
                row_widget.add_class(_row_status_class(status))
        return True

    @staticmethod
    def _button_for_action(action: ConsoleInspectorAction) -> Button:
        button = Button(
            action.label,
            id=action.widget_id,
            classes=action.classes,
            variant="primary" if action.enabled else "default",
            tooltip=action.tooltip if action.enabled else "",
        )
        button.disabled = not action.enabled
        # TASK-24606: a disabled action stays on screen, disabled. It used to
        # be given `display: none; width: 0; height: 0`, which erased both the
        # affordance and the explanation -- DESIGN.md forbids hiding why an
        # action is unavailable and names this surface -- and made action rows
        # appear and vanish between turns, costing spatial memory.
        button.styles.height = 1
        button.styles.min_height = 1
        return button

    def _compose_action(self, action: ConsoleInspectorAction) -> ComposeResult:
        yield from self._widgets_for_action(action)

    def _widgets_for_action(self, action: ConsoleInspectorAction) -> list[Widget]:
        """Build an action subtree for insertion into one bounded body."""

        widgets: list[Widget] = [self._button_for_action(action)]
        if not action.enabled and action.disabled_reason:
            # TASK-24606: rendered, not mounted-and-hidden. The reason is a
            # full sentence ("No Chatbook artifact is available."), which is
            # why it stays its own row rather than being folded into the
            # button label the way DESIGN.md's short inert-action examples
            # are -- at 33 columns "Save as Chatbook — no Chatbook artifact
            # is available." would be four wrapped rows of button.
            reason = Static(
                action.disabled_reason,
                id=f"{action.widget_id}-reason",
                classes="console-inspector-disabled-reason",
                markup=False,
            )
            widgets.append(reason)
        return widgets

    def _group_widgets(self, owner: str, *, conditional_more: bool) -> list[Widget]:
        heading_id = next(
            heading_id
            for heading, heading_id, _labels in _ROW_GROUPS
            if heading == owner
        )
        group_rows = self._rows_for(self._ownership, owner)
        group_actions = self._ownership.actions_for(owner)
        # TASK-24606 deliberately does NOT change this rule. Rendering a group
        # that holds only a disabled action would surface one more
        # explanation, but such a group contains no focusable control, and
        # `n`/`p` boundary navigation focuses the first enabled control in a
        # section -- so it would add a stop the keyboard cannot land on and
        # silently break `n` from Artifacts to Settings. That is the same
        # unfocusable-section problem TASK-24612 records as still open, and it
        # has to be solved before an empty-but-explaining group is worth
        # mounting. Groups that already render (they have rows, or an enabled
        # action) do now show their disabled actions and reasons.
        if not group_rows and not any(action.enabled for action in group_actions):
            return []
        heading = Static(
            owner,
            id=heading_id,
            classes="console-inspector-group-heading destination-section",
        )
        heading.can_focus = conditional_more and self._more_open
        body: list[Widget] = []
        for entry in group_rows:
            body.append(
                Static(
                    entry.row.text,
                    id=entry.widget_id,
                    classes=(
                        "console-inspector-row "
                        f"{_row_status_class(entry.row.status)}"
                    ),
                    markup=False,
                )
            )
        for action in group_actions:
            body.extend(self._widgets_for_action(action))
        return [
            heading,
            ConsoleBoundedSection(*body, section_id=self._section_id(owner)),
        ]

    @staticmethod
    def _section_id(owner: str) -> str:
        return owner.lower().replace(" ", "-")

    @classmethod
    def _next_ordinary_section_after_more(
        cls, ownership: InspectorOwnedContent
    ) -> str | None:
        after_conditional_block = False
        for owner, _heading_id, _labels in _ROW_GROUPS:
            if owner == "Source Readiness":
                after_conditional_block = True
                continue
            if not after_conditional_block or owner in _CONDITIONAL_OWNERS:
                continue
            if cls._rows_for(ownership, owner) or any(
                action.enabled for action in ownership.actions_for(owner)
            ):
                return cls._section_id(owner)
        return None

    def _request_sections_reconcile(self) -> None:
        """Settle every changed local body before invalidating the rail owner."""

        for section in self.query(ConsoleBoundedSection):
            section.request_reconcile()
        if self._on_reconcile is not None:
            self._on_reconcile()

    def compose(self) -> ComposeResult:
        """Compose ordinary, promoted, and More-owned Inspector groups.

        Returns:
            The current Inspector group widgets in visual order.
        """

        promoted, more = self._group_projection(self._ownership)
        for owner, _heading_id, _labels in _ROW_GROUPS:
            if owner in _CONDITIONAL_OWNERS:
                continue
            yield from self._group_widgets(owner, conditional_more=False)
            if owner == "Source Readiness":
                for promoted_owner in promoted:
                    yield from self._group_widgets(
                        promoted_owner, conditional_more=False
                    )
                more_widgets = [
                    widget
                    for more_owner in more
                    for widget in self._group_widgets(more_owner, conditional_more=True)
                ]
                if more_widgets:
                    yield ConsoleInspectorMore(
                        *more_widgets,
                        open=self._more_open,
                    )

        dict_rows = self._ownership.dictionary_rows
        dict_actions = self._ownership.dictionary_actions
        if dict_rows or dict_actions:
            yield Static(
                "Chat Dictionaries",
                id="console-inspector-dictionaries-heading",
                classes="console-inspector-group-heading destination-section",
            )
            body = []
            for entry in dict_rows:
                body.append(
                    Static(
                        entry.row.text,
                        id=entry.widget_id,
                        classes=(
                            "console-inspector-row "
                            f"console-inspector-row-{entry.row.status}"
                        ),
                        markup=False,
                    )
                )
            for entry in dict_actions:
                body.extend(self._widgets_for_action(entry.action))
            yield ConsoleBoundedSection(
                *body,
                section_id="chat-dictionaries",
            )

        world_book_rows = self._ownership.world_book_rows
        world_book_actions = self._ownership.world_book_actions
        if world_book_rows or world_book_actions:
            yield Static(
                "World Books",
                id="console-inspector-worldbooks-heading",
                classes="console-inspector-group-heading destination-section",
            )
            body = []
            for entry in world_book_rows:
                body.append(
                    Static(
                        entry.row.text,
                        id=entry.widget_id,
                        classes=(
                            "console-inspector-row "
                            f"console-inspector-row-{entry.row.status}"
                        ),
                        markup=False,
                    )
                )
            for entry in world_book_actions:
                body.extend(self._widgets_for_action(entry.action))
            yield ConsoleBoundedSection(
                *body,
                section_id="world-books",
            )
