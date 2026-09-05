"""Console status-pill strip (provider/model/assistant/RAG/source/tool/approval)
plus the retrieval-scope and cost chips.

Extracted from ConsoleControlBar so the pills can render in their own strip
directly above the composer. The widget owns the chip classes, the chip
builder, chip labelling + emphasis sync, and the approvals-review action.
"""

from __future__ import annotations

from typing import Any

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, HorizontalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
    ConsoleControlState,
    ConsoleRetrievalScopeState,
)
from tldw_chatbook.Chat.console_ephemeral import TEMPORARY_LABEL, TEMPORARY_TOOLTIP
from tldw_chatbook.Chat.rag_scope import SCOPE_REASON_EMPTY, scope_empty_notice
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Console.console_retrieval_scope_row import (
    UNSCOPED_LABEL as SCOPE_ROW_UNSCOPED_LABEL,
)


class ConsoleChip(Static):
    """Focusable Console readiness chip.

    Chips ellipsize at 22 cells; focusing a chip lifts that cap (see
    ``.console-control-chip:focus`` in ``_agentic_terminal.tcss``) so the full
    label is reachable from the keyboard, while the tooltip keeps carrying the
    same full text on hover.
    """

    can_focus = True


def _items_word(count: int) -> str:
    """Return the singular/plural scope-unit word for a tooltip count."""
    return "item" if count == 1 else "items"


class ConsoleApprovalsChip(ConsoleChip):
    """Approvals readiness chip that doubles as an approval-review action.

    Activating it (Enter/Space while focused, or click) asks the strip to
    focus the pending approval card in the transcript.
    """

    BINDINGS = [
        Binding("enter", "review_approval", "Review pending approval", show=False),
        Binding("space", "review_approval", "Review pending approval", show=False),
    ]

    class ReviewRequested(Message):
        """Posted when the approvals chip is activated from keyboard or mouse."""

    def action_review_approval(self) -> None:
        self.post_message(self.ReviewRequested())

    def _on_click(self, event: events.Click) -> None:
        self.post_message(self.ReviewRequested())


class ConsoleModelChip(ConsoleChip):
    """Provider/model chip that opens the quick model popover when activated.

    task-1670: mirrors ``ConsoleApprovalsChip``/``ConsoleScopeChip`` exactly
    -- Enter/Space while focused, or a click, opens the same popover Alt+M
    opens (``ChatScreen.action_open_console_model_popover``). Both the
    Provider and Model chips use this class; they are two views of one
    setting, so either is a reasonable place to click.
    """

    BINDINGS = [
        Binding("enter", "open_model_popover", "Open model settings", show=False),
        Binding("space", "open_model_popover", "Open model settings", show=False),
    ]

    class OpenRequested(Message):
        """Posted when a provider/model chip is activated."""

    def action_open_model_popover(self) -> None:
        self.post_message(self.OpenRequested())

    def _on_click(self, event: events.Click) -> None:
        self.post_message(self.OpenRequested())


class ConsoleAssistantChip(ConsoleChip):
    """Character/persona chip that opens the character picker when activated.

    task-1672: same activation contract as the sibling action chips. The
    chip stays actionable even when it reads "Assistant: General" -- that
    is precisely when a user most wants to pick a character.
    """

    BINDINGS = [
        Binding("enter", "open_character_picker", "Choose character", show=False),
        Binding("space", "open_character_picker", "Choose character", show=False),
    ]

    class OpenRequested(Message):
        """Posted when the assistant/character chip is activated."""

    def action_open_character_picker(self) -> None:
        self.post_message(self.OpenRequested())

    def _on_click(self, event: events.Click) -> None:
        self.post_message(self.OpenRequested())


class ConsoleSystemPromptChip(ConsoleChip):
    """System-prompt chip that opens the system-prompt editor when activated.

    Same activation contract as the sibling action chips: Enter/Space while
    focused, or a click, opens the same editor modal ``/system`` and the
    command palette open (``ChatScreen._open_console_system_prompt_editor``).
    """

    BINDINGS = [
        Binding("enter", "edit_system_prompt", "Edit system prompt", show=False),
        Binding("space", "edit_system_prompt", "Edit system prompt", show=False),
    ]

    class OpenRequested(Message):
        """Posted when the system-prompt chip is activated."""

    def action_edit_system_prompt(self) -> None:
        """Post ``OpenRequested`` when the chip is activated from the keyboard."""
        self.post_message(self.OpenRequested())

    def _on_click(self, event: events.Click) -> None:
        self.post_message(self.OpenRequested())


class ConsoleScopeChip(ConsoleChip):
    """Retrieval-scope chip that opens the scope picker when activated.

    Mirrors ``ConsoleApprovalsChip`` exactly: Enter/Space while focused, or
    a click, opens the same RAG retrieval-scope picker modal the Inspector
    row's Edit/Narrow… button opens
    (``ChatScreen._open_console_retrieval_scope_picker``, task-9) -- the
    same handler seam, task-10 just adds a second entry point into it.
    """

    BINDINGS = [
        Binding(
            "enter", "open_scope_picker", "Open retrieval scope picker", show=False
        ),
        Binding(
            "space", "open_scope_picker", "Open retrieval scope picker", show=False
        ),
    ]

    class OpenRequested(Message):
        """Posted when the scope chip is activated from keyboard or mouse."""

    def action_open_scope_picker(self) -> None:
        self.post_message(self.OpenRequested())

    def _on_click(self, event: events.Click) -> None:
        self.post_message(self.OpenRequested())


class ConsoleLibraryChip(ConsoleChip):
    """Two-axis Library policy chip that opens conversation access controls."""

    BINDINGS = [
        Binding("enter", "open_library_access", "Open Library access", show=False),
        Binding("space", "open_library_access", "Open Library access", show=False),
    ]

    class OpenRequested(Message):
        """Posted when the RAG chip is activated from keyboard or mouse."""

    def action_open_library_access(self) -> None:
        self.post_message(self.OpenRequested())

    def _on_click(self, event: events.Click) -> None:
        self.post_message(self.OpenRequested())


# Compatibility import for extensions that referenced the old class name.
# The rendered widget/id and its action now expose Library policy semantics.
ConsoleRagChip = ConsoleLibraryChip


class ConsoleSourcesChip(ConsoleChip):
    """Staged-sources chip that opens the Inspector rail when activated.

    TASK-2154.2 (DS-06/LY-11): same activation contract as the sibling
    action chips -- Enter/Space while focused, or a click. Below 150 cols
    the Inspector is the ONLY surface for staged sources (its compact
    collapse used to make this chip's content unreachable), so activation
    opens the rail itself
    (``ChatScreen._reveal_console_inspector_rail``); the staged-sources
    tray is pinned at the top of the Inspector body, visible immediately.
    """

    BINDINGS = [
        Binding("enter", "open_inspector", "Show staged sources", show=False),
        Binding("space", "open_inspector", "Show staged sources", show=False),
    ]

    class OpenRequested(Message):
        """Posted when the sources chip is activated from keyboard or mouse."""

    def action_open_inspector(self) -> None:
        """Post ``OpenRequested`` so the host screen opens the Inspector rail."""
        self.post_message(self.OpenRequested())

    def _on_click(self, event: events.Click) -> None:
        """Treat a click exactly like the Enter/Space activation."""
        self.post_message(self.OpenRequested())


class ConsoleToolsChip(ConsoleChip):
    """Tools-readiness chip that opens the Inspector rail when activated.

    TASK-2154.2 (DS-06): same activation contract as the sibling action
    chips. The run inspector (inside the Inspector rail) carries the tool
    rows, so activation opens the rail
    (``ChatScreen._reveal_console_inspector_rail``) rather than toggling
    anything in place -- "Tools: N ready" is a readout, not a latent
    switch. TASK-2154.12 (TX-04): the chip is hidden at a zero count, so
    activation is only reachable once tools are actually counted.
    """

    BINDINGS = [
        Binding("enter", "open_inspector", "Show tool readiness", show=False),
        Binding("space", "open_inspector", "Show tool readiness", show=False),
    ]

    class OpenRequested(Message):
        """Posted when the tools chip is activated from keyboard or mouse."""

    def action_open_inspector(self) -> None:
        """Post ``OpenRequested`` so the host screen opens the Inspector rail."""
        self.post_message(self.OpenRequested())

    def _on_click(self, event: events.Click) -> None:
        """Treat a click exactly like the Enter/Space activation."""
        self.post_message(self.OpenRequested())


class ConsoleRunChip(ConsoleChip):
    """Active-run chip that opens the Inspector rail when activated.

    TASK-2154.18 (FB-08): run-state copy previously had no persistent
    on-screen home between the header badge and the transcript -- the
    ``#console-mode-bar`` surface that carries ``Run: {status}`` is a
    hidden compat static. This chip is that home: visible while the
    viewed session's run status is active, hidden at idle and at terminal
    states (terminal outcomes already have their ambient signals --
    task-2154.16's failure toast, task-2154.17's success toasts, and the
    tab markers -- and a persistent "Run: complete" chip would need a
    dismissal model). Same activation contract as the sibling action
    chips: Enter/Space while focused, or a click; the Inspector's run
    rows carry the live detail, so activation opens the rail
    (``ChatScreen._reveal_console_inspector_rail``).
    """

    BINDINGS = [
        Binding("enter", "open_inspector", "Show run details", show=False),
        Binding("space", "open_inspector", "Show run details", show=False),
    ]

    class OpenRequested(Message):
        """Posted when the run chip is activated from keyboard or mouse."""

    def action_open_inspector(self) -> None:
        """Post ``OpenRequested`` so the host screen opens the Inspector rail."""
        self.post_message(self.OpenRequested())

    def _on_click(self, event: events.Click) -> None:
        """Treat a click exactly like the Enter/Space activation."""
        self.post_message(self.OpenRequested())


class ConsoleTemporaryChip(ConsoleChip):
    """Temporary-chat chip that doubles as the "Save this chat" action.

    Same activation contract as the sibling action chips: Enter/Space while
    focused, or a click. The chip is the marker AND the escape hatch, so the
    user never has to remember where saving lives.
    """

    BINDINGS = [
        Binding("enter", "save_chat", "Save this chat", show=False),
        Binding("space", "save_chat", "Save this chat", show=False),
    ]

    class SaveRequested(Message):
        """Posted when the temporary chip is activated."""

    def action_save_chat(self) -> None:
        self.post_message(self.SaveRequested())

    def _on_click(self, event: events.Click) -> None:
        self.post_message(self.SaveRequested())


class ConsoleCostChip(ConsoleChip):
    """Cost chip that opens the cost breakdown modal when activated (task-4).

    Same activation contract as the sibling action chips: Enter/Space while
    focused, or a click. The chip is the running-total ticker AND the entry
    point into the per-message breakdown, so a user who wants to know why
    the total looks the way it does never has to hunt for it elsewhere.
    """

    BINDINGS = [
        Binding("enter", "open_cost_breakdown", "Open cost breakdown", show=False),
        Binding("space", "open_cost_breakdown", "Open cost breakdown", show=False),
    ]

    class ConsoleCostChipPressed(Message):
        """Posted when the cost chip is activated from keyboard or mouse."""

    def action_open_cost_breakdown(self) -> None:
        """Post the activation event that opens the cost breakdown modal."""
        self.post_message(self.ConsoleCostChipPressed())

    def _on_click(self, event: events.Click) -> None:
        self.post_message(self.ConsoleCostChipPressed())


class ConsoleStatusChips(Horizontal):
    """Full-width strip of Console readiness pills (provider/model/assistant/
    RAG/source/tool/approval plus the retrieval-scope chip).

    TASK-2154.5 (LY-03): the expanded presentation's inner strip scrolls
    horizontally when the chips outgrow the viewport instead of silently
    clipping them -- keyboard reachability comes from focus auto-scroll
    (``Screen.set_focus``'s ``scroll_visible``), mouse from Shift+wheel /
    trackpad swipe; the scrollbar itself is hidden (single-row strip,
    tab-strip precedent).

    Exposes ``sync_state`` so ``ChatScreen`` can refresh the pill labels and
    counter emphasis after provider/model/source/tool/approval state changes.
    """

    def __init__(
        self,
        state: ConsoleControlState,
        *,
        scope_state: ConsoleRetrievalScopeState | None = None,
        ephemeral: bool = False,
        cost_state: ConsoleCostState | None = None,
        run_copy: str = "",
        collapsed: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the strip.

        Args:
            state: Display-state snapshot for the readiness labels.
            scope_state: Display-state snapshot for the "Scope" chip
                (task-10) -- the same ``ConsoleRetrievalScopeState`` the
                Inspector's retrieval-scope row renders from. ``None``
                renders as unscoped (hidden).
            ephemeral: Whether the active session is temporary at
                construction time (final-review F1). Without this the
                Temporary chip always composed as "not temporary" on a
                freshly (re)constructed screen -- e.g. after Console ->
                another screen -> Console navigation, which builds a brand
                new ``ChatScreen``/``ConsoleStatusChips`` -- and stayed
                wrong until something happened to call
                ``sync_temporary_chip`` by hand. Callers should still call
                ``ChatScreen._sync_console_temporary_chip()`` after mount
                as a second line of defense for session switches that
                happen post-construction.
            cost_state: Display-state snapshot for the cost chip (task-4,
                PR3 cost ticker) -- mirrors the F1 precedent above: passed
                in at construction so the chip renders correctly on the
                very first frame rather than waiting for a post-mount
                ``sync_cost_state`` call. ``None`` renders hidden (non-
                Console-native contexts have no cost to show).
            run_copy: Active-run copy for the run chip (TASK-2154.18,
                FB-08) -- same F1 precedent: returning to Console while a
                background run is still streaming must render the chip on
                the first frame, not after the next sync tick. ``""``
                (or any non-active state) renders hidden.
            collapsed: Whether to show the one-line collapsed presentation.
            **kwargs: Additional Textual widget arguments (id/classes).
        """
        classes = kwargs.pop("classes", "")
        # Reuse the existing chip-row class so its CSS continues to apply.
        super().__init__(
            classes=f"console-control-chip-row console-status-chips {classes}".strip(),
            **kwargs,
        )
        self.state = state
        self.scope_state = scope_state
        self.ephemeral = ephemeral
        self._cost_state = cost_state
        self._run_chip_state: tuple[bool, str] = (bool(run_copy), run_copy)
        self._collapsed = bool(collapsed)
        self.styles.height = 1
        self.styles.min_height = 1
        self.styles.max_height = 1

    @staticmethod
    def _chip(
        label: str,
        *,
        id: str,
        emphasis: bool | None = None,
        chip_class: type[ConsoleChip] = ConsoleChip,
    ) -> ConsoleChip:
        """Build one readiness chip. Mirrors the former ConsoleControlBar._chip."""
        classes = "console-control-chip"
        if emphasis is False:
            classes += " console-chip-dim"
        elif emphasis is True:
            classes += " console-chip-alert"
        # markup=False: chip labels carry user data (assistant names and model
        # ids). A name containing `[red]...[/]` would otherwise
        # restyle the chip strip, or raise MarkupError when unbalanced.
        chip = chip_class(label, id=id, classes=classes, markup=False)
        chip.tooltip = Content(label)
        return chip

    def compose(self) -> ComposeResult:
        expanded = Horizontal(id="console-status-expanded")
        expanded.styles.display = "none" if self._collapsed else "block"
        with expanded:
            collapse_button = Button(
                "Status ▾",
                id="console-status-collapse",
                compact=True,
            )
            collapse_button.tooltip = "Collapse status details."
            collapse_button.styles.width = 9
            collapse_button.styles.min_width = 9
            collapse_button.styles.max_width = 9
            collapse_button.styles.line_pad = 0
            yield collapse_button
            chip_scroll = HorizontalScroll(id="console-status-chip-scroll")
            chip_scroll.styles.height = 1
            chip_scroll.styles.min_height = 1
            chip_scroll.styles.max_height = 1
            chip_scroll.styles.scrollbar_size_horizontal = 0
            with chip_scroll:
                # First: this is a property of the whole chat, not one setting.
                yield self._temporary_chip()
                # TASK-2154.18 (FB-08): the active-run chip sits left-most among
                # the transient chips so it stays visible when the strip scrolls
                # horizontally (TASK-2154.5) -- the stable readiness chips keep
                # their learned relative order behind it. Hidden unless a run is
                # active (see ``sync_run_chip``).
                yield self._run_chip()
                # The two-axis Library policy is the primary permission
                # readout. Keep it ahead of provider/model metadata so both
                # axes remain painted before horizontal overflow begins.
                yield self._chip(
                    self.state.rag_label,
                    id="console-library-chip",
                    chip_class=ConsoleLibraryChip,
                )
                yield self._chip(
                    self.state.provider_label,
                    id="console-provider-chip",
                    chip_class=ConsoleModelChip,
                )
                yield self._chip(
                    self.state.model_label,
                    id="console-model-chip",
                    chip_class=ConsoleModelChip,
                )
                yield self._chip(
                    self.state.system_prompt_label,
                    id="console-system-prompt-chip",
                    chip_class=ConsoleSystemPromptChip,
                )
                yield self._chip(
                    self.state.assistant_label,
                    id="console-assistant-chip",
                    chip_class=ConsoleAssistantChip,
                )
                yield self._chip(
                    self.state.sources_label,
                    id="console-sources-chip",
                    emphasis=self.state.sources_active,
                    chip_class=ConsoleSourcesChip,
                )
                tools_chip = self._chip(
                    self.state.tools_label,
                    id="console-tools-chip",
                    emphasis=self.state.tools_active,
                    chip_class=ConsoleToolsChip,
                )
                # TASK-2154.12 (TX-04): hidden entirely at a zero tool count, the
                # same posture as the unscoped scope chip and the None cost chip --
                # the old "Tools: not loaded" placeholder exposed a lazy-loading
                # implementation detail (Console UX review 2026-08).
                tools_chip.display = self.state.tools_active
                yield tools_chip
                yield self._chip(
                    self.state.approvals_label,
                    id="console-approvals-chip",
                    emphasis=self.state.approvals_active,
                    chip_class=ConsoleApprovalsChip,
                )
                # task-10: the retrieval-scope chip -- unlike the chips above,
                # hidden entirely when unscoped rather than showing a
                # "Scope: everything" default (see ``_scope_chip_render``).
                yield self._scope_chip()
                # task-4 (PR3 cost ticker): last in the strip, hidden entirely
                # when there is no cost state (see ``_cost_chip_render``).
                yield self._cost_chip()

        collapsed = Horizontal(id="console-status-collapsed")
        collapsed.styles.display = "block" if self._collapsed else "none"
        with collapsed:
            expand_button = Button(
                "Status ▴",
                id="console-status-expand",
                compact=True,
            )
            expand_button.tooltip = "Expand status details."
            expand_button.styles.width = 9
            expand_button.styles.min_width = 9
            expand_button.styles.max_width = 9
            expand_button.styles.line_pad = 0
            yield expand_button
            yield Static("Status hidden", id="console-status-collapsed-copy")

    @property
    def collapsed(self) -> bool:
        """Whether the collapsed status presentation is active."""
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Toggle the mounted status presentations without touching chip state."""
        self._collapsed = bool(collapsed)
        self.query_one(
            "#console-status-expanded", Horizontal
        ).display = not self._collapsed
        self.query_one(
            "#console-status-collapsed", Horizontal
        ).display = self._collapsed

    def _run_chip(self) -> ConsoleRunChip:
        label, tooltip, hidden = self._run_chip_render(*self._run_chip_state)
        chip = self._chip(
            label,
            id="console-run-chip",
            chip_class=ConsoleRunChip,
        )
        chip.tooltip = tooltip
        chip.display = not hidden
        return chip

    @staticmethod
    def _run_chip_render(visible: bool, copy: str) -> tuple[str, str, bool]:
        """Pure ``(label, tooltip, hidden)`` render for the run chip.

        Args:
            visible: Whether the viewed session's run status is active
                (the screen owns the ``CONSOLE_ACTIVE_RUN_STATUSES``
                membership call; the widget stays display-only).
            copy: The run's ``visible_copy`` (e.g. "Streaming response.").
        """
        if not visible:
            return "", "", True
        text = copy.strip() or "Working."
        return (
            f"Run: {text}",
            f"Active run: {text} Open the Inspector for run details.",
            False,
        )

    def sync_run_chip(self, visible: bool, copy: str) -> None:
        """Refresh the run chip's visibility and copy (TASK-2154.18, FB-08).

        Deliberately NOT folded into ``sync_state``: run state changes on
        its own cadence (send/stop transitions and the 0.2s transcript
        poll while a run is active), independently of
        ``ConsoleControlState`` -- same reason ``sync_cost_state`` and
        ``sync_scope_chip`` stand alone. Equality-guarded so the poll
        ticks are free when nothing changed.

        Args:
            visible: Whether the viewed session's run is in an active
                status. Terminal states hide the chip (their ambient
                signals live elsewhere -- see ``ConsoleRunChip``).
            copy: The run's ``visible_copy``; ignored when not visible.
        """
        next_state = (visible, copy if visible else "")
        if next_state == self._run_chip_state:
            return
        self._run_chip_state = next_state
        try:
            chip = self.query_one("#console-run-chip", ConsoleRunChip)
        except NoMatches:
            return
        label, tooltip, hidden = self._run_chip_render(*next_state)
        if hidden:
            chip.display = False
            return
        chip.update(label)
        chip.tooltip = Content(tooltip)
        chip.display = True

    def _temporary_chip(self) -> ConsoleTemporaryChip:
        label, tooltip, hidden = self._temporary_chip_render(self.ephemeral)
        chip = self._chip(
            label,
            id="console-temporary-chip",
            chip_class=ConsoleTemporaryChip,
        )
        chip.tooltip = tooltip
        chip.display = not hidden
        return chip

    @staticmethod
    def _temporary_chip_render(ephemeral: bool) -> tuple[str, str, bool]:
        """Pure ``(label, tooltip, hidden)`` render for the temporary chip.

        Args:
            ephemeral: Whether the active session is temporary.

        Returns:
            ``label``: chip text. ``tooltip``: hover/focus text, which is
            where the save affordance is spelled out. ``hidden``: ``True``
            for a normal chat -- a "Saved" chip on every ordinary
            conversation would be noise, and the strip is width-bounded.
        """
        if not ephemeral:
            return TEMPORARY_LABEL, TEMPORARY_TOOLTIP, True
        return TEMPORARY_LABEL, TEMPORARY_TOOLTIP, False

    def sync_temporary_chip(self, ephemeral: bool) -> None:
        """Refresh the temporary chip from the active session's flag.

        Separate from ``sync_state`` for the same reason ``sync_scope_chip``
        is: this is pushed from the screen when the active session changes,
        not on every control-bar sync tick.

        Args:
            ephemeral: Whether the active session is temporary.
        """
        if ephemeral == self.ephemeral:
            return
        self.ephemeral = ephemeral
        try:
            chip = self.query_one("#console-temporary-chip", ConsoleTemporaryChip)
        except NoMatches:
            return
        label, tooltip, hidden = self._temporary_chip_render(ephemeral)
        chip.update(label)
        chip.tooltip = tooltip
        chip.display = not hidden

    def _scope_chip(self) -> ConsoleScopeChip:
        label, tooltip, hidden, alert = self._scope_chip_render(self.scope_state)
        chip = self._chip(
            label,
            id="console-scope-chip",
            emphasis=True if alert else None,
            chip_class=ConsoleScopeChip,
        )
        chip.tooltip = tooltip
        chip.display = not hidden
        return chip

    @staticmethod
    def _scope_chip_render(
        state: ConsoleRetrievalScopeState | None,
    ) -> tuple[str, str, bool, bool]:
        """Pure ``(label, tooltip, hidden, alert)`` render for the scope chip.

        ``item_count`` is always the EFFECTIVE (post-intersection) count
        (task-13). The tooltip spells the active scope levels out in words
        (TASK-2154.12/TX-03 -- the old ``conversation A ∩ workspace B → N``
        math notation and the raw ``scope_empty`` cause token are gone): a
        single active level (conversation-only, or workspace-only) reads as
        "Only searching: conversation scope (N items)"/"Only searching:
        workspace scope (N items)"; both active levels read as the full
        breakdown with the shared count named in words.

        Args:
            state: Display-state snapshot, or ``None`` (renders unscoped).

        Returns:
            ``label``: chip text. ``tooltip``: hover/focus text (the
            EMPTY branch folds the plain-language cause in). ``hidden``:
            ``True`` when unscoped (chip carries no useful information --
            hidden rather than shown as "everything", matching the brief).
            ``alert``: ``True`` only for EMPTY, reusing the same
            ``console-chip-alert`` action-required styling the
            sources/tools/approvals chips use when their own count is
            active.
        """
        if state is None or (not state.is_scoped and not state.is_empty):
            return SCOPE_ROW_UNSCOPED_LABEL, "", True, False
        if state.is_empty:
            return (
                "Scope: no sources",
                scope_empty_notice(state.cause or SCOPE_REASON_EMPTY),
                False,
                True,
            )
        label = f"Scope: {state.item_count}"
        if state.conv_item_count is not None and state.ws_item_count is not None:
            tooltip = (
                f"Only searching: conversation scope "
                f"({state.conv_item_count} {_items_word(state.conv_item_count)}) "
                f"and workspace scope "
                f"({state.ws_item_count} {_items_word(state.ws_item_count)}) — "
                f"{state.item_count} in both."
            )
        elif state.ws_item_count is not None:
            tooltip = (
                f"Only searching: workspace scope "
                f"({state.ws_item_count} {_items_word(state.ws_item_count)})."
            )
        else:
            tooltip = (
                f"Only searching: conversation scope "
                f"({state.item_count} {_items_word(state.item_count)})."
            )
        return label, tooltip, False, False

    def sync_scope_chip(self, scope_state: ConsoleRetrievalScopeState | None) -> None:
        """Refresh the "Scope" chip from a new snapshot (task-10).

        Deliberately NOT folded into ``sync_state`` above: this is pushed
        directly from ``ChatScreen._sync_console_retrieval_scope_row`` with
        the exact same ``ConsoleRetrievalScopeState`` instance passed to the
        Inspector row's own ``sync_state`` in the same call -- one state,
        two renderers, computed once. Keeping it a separate method also
        keeps the chip's refresh triggers identical to the row's: the
        general ``sync_state`` refresh (called far more often, e.g. every
        control-bar sync tick) never touches this chip at all.

        Args:
            scope_state: Updated display-state snapshot to render.
        """
        if scope_state == self.scope_state:
            return
        self.scope_state = scope_state
        try:
            chip = self.query_one("#console-scope-chip", ConsoleScopeChip)
        except NoMatches:
            return
        label, tooltip, hidden, alert = self._scope_chip_render(scope_state)
        chip.update(label)
        chip.tooltip = tooltip
        chip.display = not hidden
        chip.set_class(alert, "console-chip-alert")

    def _cost_chip(self) -> ConsoleCostChip:
        label, tooltip, hidden, alert, cold = self._cost_chip_render(self._cost_state)
        # ``_chip``'s emphasis only knows dim/alert -- cold is a third,
        # non-alarming state added on top (see the class-toggle table in
        # ``sync_cost_state``). Neutral (``None``) suppresses both so the
        # cold class is the only one applied.
        emphasis: bool | None
        if alert:
            emphasis = True
        elif cold:
            emphasis = None
        else:
            emphasis = False
        chip = self._chip(
            label,
            id="console-cost-chip",
            emphasis=emphasis,
            chip_class=ConsoleCostChip,
        )
        chip.tooltip = Content(tooltip)
        chip.display = not hidden
        if cold:
            chip.add_class("console-chip-cold")
        return chip

    @staticmethod
    def _cost_chip_render(
        state: ConsoleCostState | None,
    ) -> tuple[str, str, bool, bool, bool]:
        """Pure ``(label, tooltip, hidden, alert, cold)`` render for the cost chip.

        Args:
            state: Display-state snapshot, or ``None`` (hides the chip --
                non-Console-native contexts have no cost to show).

        Returns:
            ``label``: chip text (the full ``state.label``; the strip
            hasn't been laid out yet at compose time, so the width-aware
            compact form applies on the first resize or ``sync_cost_state``).
            ``tooltip``: hover/focus text. ``hidden``: ``True`` when
            ``state`` is ``None``. ``alert``/``cold``: ``state.alert``/
            ``state.cold``.
        """
        if state is None:
            return "", "", True, False, False
        return state.label, state.tooltip, False, state.alert, state.cold

    def sync_cost_state(self, state: ConsoleCostState | None) -> None:
        """Refresh the cost chip from a new cost-state snapshot (task-4).

        Deliberately NOT folded into ``sync_state`` below, for the same
        reason ``sync_scope_chip`` isn't: cost state is computed and
        pushed on its own cadence (the general control-bar sync tick, but
        independently of whether ``ConsoleControlState`` itself changed),
        so it owns its own equality guard.

        Args:
            state: Updated cost-chip snapshot, or ``None`` to hide the
                chip.
        """
        if state == self._cost_state:
            return
        self._cost_state = state
        try:
            chip = self.query_one("#console-cost-chip", ConsoleCostChip)
        except NoMatches:
            return
        if state is None:
            chip.display = False
            return
        self._apply_cost_chip_label(chip)
        chip.tooltip = Content(state.tooltip)
        chip.display = True
        chip.set_class(state.alert, "console-chip-alert")
        chip.set_class(state.cold, "console-chip-cold")
        chip.set_class(not state.alert and not state.cold, "console-chip-dim")

    def _apply_cost_chip_label(self, chip: ConsoleCostChip) -> None:
        """Apply the current cost state's full or compact width-aware label."""
        state = self._cost_state
        if state is None:
            return
        label = state.compact_label if self.screen.size.width < 120 else state.label
        chip.update(label)
        chip.refresh(layout=True)

    def on_resize(self, _event: events.Resize) -> None:
        """Reapply width-aware cost copy without requiring a state change."""
        try:
            chip = self.query_one("#console-cost-chip", ConsoleCostChip)
        except NoMatches:
            return
        self._apply_cost_chip_label(chip)

    def sync_state(self, state: ConsoleControlState) -> None:
        """Refresh pill labels and counter emphasis from a new snapshot."""
        if state == self.state:
            return
        self.state = state
        label_values = {
            "#console-provider-chip": state.provider_label,
            "#console-model-chip": state.model_label,
            "#console-system-prompt-chip": state.system_prompt_label,
            "#console-assistant-chip": state.assistant_label,
            "#console-library-chip": state.rag_label,
            "#console-sources-chip": state.sources_label,
            "#console-tools-chip": state.tools_label,
            "#console-approvals-chip": state.approvals_label,
        }
        for selector, label in label_values.items():
            try:
                chip = self.query_one(selector, Static)
            except NoMatches:
                continue
            chip.update(label)
            chip.tooltip = Content(label)
        chip_emphasis = {
            "#console-sources-chip": state.sources_active,
            "#console-tools-chip": state.tools_active,
            "#console-approvals-chip": state.approvals_active,
        }
        for selector, active in chip_emphasis.items():
            try:
                chip = self.query_one(selector, Static)
            except NoMatches:
                continue
            chip.set_class(not active, "console-chip-dim")
            chip.set_class(active, "console-chip-alert")
        # TASK-2154.12 (TX-04): the tools chip hides at a zero count (see
        # compose); keep its visibility in step on every sync.
        try:
            tools_chip = self.query_one("#console-tools-chip", Static)
        except NoMatches:
            pass
        else:
            tools_chip.display = state.tools_active

    @on(ConsoleApprovalsChip.ReviewRequested)
    def on_approval_review_requested(
        self, event: ConsoleApprovalsChip.ReviewRequested
    ) -> None:
        """Focus the pending approval card in the transcript.

        Falls back to a notification when no approval is pending so the
        keyboard-only path never dead-ends silently.
        """
        event.stop()
        self._focus_pending_approval_card()

    def _focus_pending_approval_card(self) -> None:
        """Scroll the displayed approval card into view and focus its action."""
        try:
            cards = list(self.screen.query("#chat-approval-card"))
        except Exception:
            cards = []
        card = next(
            (
                candidate
                for candidate in cards
                if isinstance(candidate, ChatApprovalCard) and candidate.display
            ),
            None,
        )
        if card is None:
            self.app.notify(CONSOLE_INSPECTOR_NO_APPROVAL_REASON, severity="warning")
            return
        try:
            card.scroll_visible(animate=False)
        except Exception:
            pass
        # `set_batch` (the card's sole production entry point, task-914) is
        # the only body it ever renders, so a displayed card's action is
        # always its "Submit" button.
        try:
            card.focus_first_decision()
        except NoMatches:
            pass
