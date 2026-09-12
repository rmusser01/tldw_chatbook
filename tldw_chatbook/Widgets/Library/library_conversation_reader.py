"""Permanent read-only work pane for saved Library conversations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.message import Message
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_conversation_reader_state import (
    ConversationMessageView,
    ConversationReaderState,
)
from tldw_chatbook.Library.library_shell_state import library_disabled_action_label
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    AdaptiveReaderShellResized,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    format_console_relative_age,
)


def _open_console_disabled_tooltip(state: ConversationReaderState) -> str | None:
    """Describe the current reason the retained transcript cannot be opened."""
    if state.loaded_actions_eligible:
        return None
    if state.bulk_active:
        return "Finish bulk selection before opening a conversation in Console."
    if state.loading:
        return "Wait for the selected conversation to finish loading."
    if state.unavailable:
        return "The selected conversation is unavailable."
    if state.error:
        return "Try again before opening the selected conversation in Console."
    if (
        state.selected_id != state.loaded_id
        or state.selected_version != state.loaded_version
        or state.generation != state.loaded_generation
    ):
        return "The selected conversation does not match the retained transcript."
    return "Wait for the complete selected transcript before opening it in Console."


def library_conversation_link_would_unblock(
    state: ConversationReaderState,
    loaded_metadata: Mapping[str, Any],
) -> bool:
    """Return whether a workspace link is what would unblock this hand-off.

    (task-32107) The ONE rule behind both the reader's link affordance and
    the screen handler's decision to link before staging: the button's
    enabled state and the press's behaviour cannot disagree because they ask
    the same question of the same three inputs.

    Fenced by ``loaded_actions_eligible`` (review round 2 of task-32056): the
    link writes membership for the RETAINED ``loaded_id``, so while a newly
    selected conversation is still loading, the visible-but-stale transcript
    would otherwise be the one linked.

    Args:
        state: The reader state, whose load fence answers first.
        loaded_metadata: The controller-injected metadata carrying
            ``_workspace_block`` and ``_workspace_block_linkable``.

    Returns:
        True when a link into the active workspace resolves the only block.
    """
    return (
        state.loaded_actions_eligible
        and bool(str(loaded_metadata.get("_workspace_block") or "").strip())
        and bool(loaded_metadata.get("_workspace_block_linkable"))
    )


def library_conversation_block_sentence(
    state: ConversationReaderState,
    *,
    blocked: str,
    detail: str = "",
    link_offered: bool = False,
) -> str | None:
    """Return the one sentence that explains a refused Console hand-off.

    (task-32101) The single source for all three surfaces that state the
    refusal: the disabled action's tooltip, the reason line beneath it, and
    the ``c`` accelerator's toast. They used to be three separate strings,
    which is how the line under the button ended up repeating the action
    name the button already carried and the key ended up saying nothing at
    all.

    Args:
        state: The reader state, whose load fence answers first -- while it
            holds, the workspace block is not the reason the press is
            unavailable.
        blocked: The short workspace refusal phrase ("not in this
            workspace"), or empty when the conversation is eligible.
        detail: The eligibility rule's own recovery sentence, used for the
            blocks a link cannot resolve.
        link_offered: Whether "Link to workspace" is actually on screen; the
            remedy is only NAMED when it is.

    Returns:
        The sentence, or ``None`` when the hand-off may run.
    """
    load_block = _open_console_disabled_tooltip(state)
    if load_block:
        return load_block
    if not blocked:
        return None
    if link_offered:
        # task-32107 (user decision, critique #10): the primary action no
        # longer refuses a block a link can resolve -- it links and proceeds
        # -- so the line says what the press will DO rather than sending the
        # reader to a second button. Membership decides what a Console turn
        # may read, so the widening is still stated before it happens. The
        # action's NAME stays on the action (task-32101 AC#4): this line
        # points at the control above it, it does not re-label it.
        return (
            f"This conversation is {blocked}. Pressing this adds it to the "
            "active workspace first, and you can undo that."
        )
    return detail or f"This conversation is {blocked}."


class LibraryConversationReader(Vertical):
    """Render one retained Conversations Read/Info pane from pure state."""

    class MessagesSynced(Message):
        """Notify the controller that current transcript rows are mounted."""

        def __init__(self, reader_generation: int, find_query: str) -> None:
            self.reader_generation = reader_generation
            self.find_query = find_query
            super().__init__()

    def __init__(
        self,
        state: ConversationReaderState,
        *,
        metadata: Mapping[str, Any] | None = None,
        loaded_metadata: Mapping[str, Any] | None = None,
        selected_metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.loaded_metadata = dict(loaded_metadata or metadata or {})
        self.selected_metadata = dict(selected_metadata or {})
        self._message_sync_generation = 0
        self._find_navigation_key = None
        self._find_navigation_index = -1
        self._pending_find_navigation_index: int | None = None

    def _workspace_block(self) -> str:
        """Return the short workspace refusal reason, or empty when eligible.

        Injected by the controller alongside the other computed metadata
        keys (``_list_status``/``_list_summary``): the reason is derived
        from the workspace registry, which this pure widget never reads.
        """
        return str(self.loaded_metadata.get("_workspace_block") or "").strip()

    def _workspace_block_detail(self) -> str:
        """Return the eligibility rule's own sentence for this block, if any."""
        return str(self.loaded_metadata.get("_workspace_block_detail") or "").strip()

    def _workspace_link_offered(self) -> bool:
        """Whether a workspace link would actually resolve the block."""
        return library_conversation_link_would_unblock(
            self.state, self.loaded_metadata
        )

    def _actions_enabled(self) -> bool:
        """Whether the Console hand-off may run with nothing else happening."""
        return self.state.loaded_actions_eligible and not self._workspace_block()

    def _source_press_enabled(self) -> bool:
        """Whether "Use as source" may be pressed at all.

        (task-32107, user decision) A block a LINK can resolve no longer
        disables the hand-off -- pressing it links the conversation into the
        active workspace and proceeds, in one undoable step. The load fence
        and the blocks a link cannot resolve still disable it.
        """
        return self._actions_enabled() or self._workspace_link_offered()

    def _workspace_link_receipt(self) -> str:
        """Return the workspace the last press linked into, or empty.

        (re-review P3) Withheld while ANY workspace block stands, because the
        receipt's whole claim is "this conversation can now be used in
        Console". Activating a workspace this conversation is not in brings
        the block straight back, and the reader would otherwise paint the
        refusal sentence and a receipt contradicting it, one under the other.
        The block is recomputed on every sync, so this needs no state of its
        own -- and the receipt reappears untouched if that workspace is made
        active again.
        """
        if self._workspace_block():
            return ""
        return str(self.loaded_metadata.get("_workspace_link_receipt") or "").strip()

    def _source_label(self) -> str:
        """Return the hand-off button label, marked when it is refused.

        (task-32101) The non-colour disabled marker belongs on the control
        that carries the action name, so the reason line beneath it need not
        repeat that name to carry the marker. It follows the button's real
        ``disabled`` state (task-32107): a pressable action never wears the
        "○" the glyph legend reserves for a blocked one.
        """
        return library_disabled_action_label(
            "Use as source", not self._source_press_enabled()
        )

    def _blocked_reason_line(self) -> str:
        """Return the wrapping reason line shown under a blocked action.

        (fix round 1) The reason used to live in the ``Button`` label,
        which Textual renders on a single truncating line -- at 100x30 the
        Conversations reader pane is ~43 cells and it clipped to "...not in
        this worksp". A ``Static`` wraps, so the copy survives every width.

        (task-32101) It is the refusal SENTENCE, not a second copy of the
        action name: repeating "○ Open in Console" under the button that
        already says it read as two controls. The disabled marker moved onto
        the button label, where the one action name lives.
        """
        if not self._workspace_block():
            return ""
        return self._open_console_tooltip() or ""

    def _open_console_tooltip(self) -> str | None:
        """Return the current reason the hand-off cannot run.

        (review round 2) The load fence answers first -- while it holds, the
        workspace block is not the reason the press is unavailable -- and the
        remedy is only NAMED when it is actually on screen. A block linking
        cannot resolve keeps the eligibility rule's own recovery sentence
        ("Select an active workspace...") instead of pointing at a hidden
        button. (task-32101) The rule itself lives in
        ``library_conversation_block_sentence``, shared with the screen's
        ``c`` accelerator so the key cannot say something else.
        """
        return library_conversation_block_sentence(
            self.state,
            blocked=self._workspace_block(),
            detail=self._workspace_block_detail(),
            link_offered=self._workspace_link_offered(),
        )

    def compose(self) -> ComposeResult:
        """Compose stable controls and the initially available transcript."""
        yield Static("Conversation reader", classes="destination-section", markup=False)
        with Horizontal(classes="ds-toolbar library-conversation-reader-modes"):
            read = Button(
                "Read",
                id="library-conversation-reader-read",
                classes="library-canvas-action",
                compact=True,
            )
            read.set_class(self.state.mode == "read", "-selected")
            yield read
            info = Button(
                "Info",
                id="library-conversation-reader-info",
                classes="library-canvas-action",
                compact=True,
            )
            info.set_class(self.state.mode == "info", "-selected")
            yield info
        # (task-32056) The hand-off is header chrome, beside Read/Info,
        # where Media puts "Use in Console" -- it used to sit at the very
        # bottom of the pane, below a 30-message transcript. Stacked, not
        # a row: the blocked label carries its reason, and at 100 and 60
        # columns a row clipped both it and the remedy beside it (live
        # verification) -- the exact failure this task exists to close.
        with Vertical(classes="ds-toolbar library-conversation-reader-actions"):
            open_console = Button(
                "Restore and resume"
                if (
                    self.loaded_metadata.get("archived")
                    or self.loaded_metadata.get("workspace_archived")
                )
                else "Resume conversation",
                id="library-conversation-open-console",
                classes="library-canvas-action",
                compact=True,
            )
            open_console.disabled = not self.state.loaded_actions_eligible
            open_console.tooltip = _open_console_disabled_tooltip(self.state)
            yield open_console
            source = Button(
                self._source_label(),
                id="library-conversation-use-source",
                classes="library-canvas-action",
                compact=True,
            )
            source.disabled = not self._source_press_enabled()
            source.tooltip = self._open_console_tooltip()
            yield source
            # (task-32107 review, fix round 1) Directly under the control it
            # describes. The sentence says "Pressing this", so its position is
            # load-bearing: yielded after the Archive/Restore pair it sat
            # under "Archive conversation" and the deixis pointed at the
            # wrong button. ``sync_state`` patches by id, so only this order
            # matters.
            blocked_reason = Static(
                self._blocked_reason_line(),
                id="library-conversation-open-console-blocked",
                classes="library-conversation-reader-block-reason",
                markup=False,
            )
            # (fix round 1) Visibility and content come from ONE predicate:
            # the line answers the load fence first, so gating display on
            # the workspace block alone showed "Wait for the selected
            # conversation to finish loading." under a button that was on
            # screen for a different reason.
            blocked_reason.display = bool(self._blocked_reason_line())
            yield blocked_reason
            for action, label in (
                ("archive", "Archive conversation"),
                ("restore", "Restore only"),
            ):
                button = Button(
                    label,
                    id=f"library-conversation-{action}",
                    classes="library-canvas-action",
                    compact=True,
                )
                button.display = bool(self.loaded_metadata.get("archived")) == (
                    action == "restore"
                )
                button.disabled = not self.state.loaded_actions_eligible
                yield button
            link = Button(
                "Link to workspace",
                id="library-conversation-link-workspace",
                classes="library-canvas-action",
                compact=True,
            )
            link.display = self._workspace_link_offered()
            yield link
            # (task-32107) The receipt for the membership "Use as source"
            # just wrote, and its inverse. Workspace membership decides what
            # a Console turn may read, so the widening is a visible,
            # reversible act rather than a silent side effect.
            receipt_workspace = self._workspace_link_receipt()
            receipt = Static(
                f"✓ linked · {receipt_workspace} · this conversation can now "
                "be used in Console",
                id="library-conversation-link-receipt",
                classes="library-conversation-reader-block-reason",
                markup=False,
            )
            receipt.display = bool(receipt_workspace)
            yield receipt
            undo = Button(
                "Undo link",
                id="library-conversation-link-undo",
                classes="library-canvas-action",
                compact=True,
            )
            undo.display = bool(receipt_workspace)
            yield undo
            retry = Button(
                "Try again",
                id="library-conversation-reader-retry",
                classes="library-canvas-action",
                compact=True,
            )
            retry.display = bool(self.state.error or self.state.unavailable)
            yield retry
        yield Static(
            self._status_text(),
            id="library-conversation-reader-status",
            markup=False,
        )
        find = Input(
            value=self.state.find_query,
            placeholder="Find in complete transcript…",
            id="library-conversation-reader-find",
        )
        find.display = self.state.mode == "read"
        yield find
        with Horizontal(
            classes="ds-toolbar", id="library-conversation-find-navigation"
        ):
            for direction, label in (
                ("previous", "Find previous"),
                ("next", "Find next"),
            ):
                yield Button(
                    label,
                    id=f"library-conversation-find-{direction}",
                    compact=True,
                    disabled=not self.state.find_matches,
                )
        yield Static("", id="library-conversation-find-position", markup=False)
        messages = VerticalScroll(id="library-conversation-reader-messages")
        messages.display = self.state.mode == "read"
        with messages:
            for message in self.state.messages:
                yield self._message_widget(message)
        info_body = Static(
            self._metadata_text(),
            id="library-conversation-reader-info-body",
            markup=False,
        )
        info_body.display = self.state.mode == "info"
        yield info_body

    def on_mount(self) -> None:
        """Project initial labels and visibility without replacing this widget."""
        self.sync_state(
            self.state,
            loaded_metadata=self.loaded_metadata,
            selected_metadata=self.selected_metadata,
        )

    @staticmethod
    def _message_copy(message: ConversationMessageView) -> str:
        # task-32067: the same compact age the Conversations list shows
        # ("27m", "2d"), not the stored ISO stamp the transcript used to
        # repeat above every message. Unparseable stamps format to "", and
        # the join below then drops the slot entirely.
        age = format_console_relative_age(
            message.timestamp, now=datetime.now(timezone.utc)
        )
        heading = " · ".join(value for value in (message.sender, age) if value)
        return f"{heading}\n{message.text}" if heading else message.text

    @classmethod
    def _message_widget(cls, message: ConversationMessageView) -> Static:
        row = Static(
            cls._message_copy(message),
            classes="library-conversation-reader-message",
            markup=False,
        )
        row.message_id = message.message_id
        row.can_focus = True
        return row

    def _metadata_text(self) -> str:
        loaded = self.state.loaded_id is not None
        title = str(self.loaded_metadata.get("title") or "Unknown title")
        conversation_id = self.state.loaded_id or "unknown"
        version = self.state.loaded_version
        workspace = str(
            self.loaded_metadata.get("workspace")
            or self.loaded_metadata.get("workspace_name")
            or "unassigned"
        )
        updated = str(
            self.loaded_metadata.get("last_modified")
            or self.loaded_metadata.get("updated_at")
            or self.loaded_metadata.get("updated")
            or "unknown"
        )
        raw_keywords = self.loaded_metadata.get("keywords")
        if isinstance(raw_keywords, (list, tuple)):
            keywords = ", ".join(str(value) for value in raw_keywords if str(value))
        else:
            keywords = ""
        return "\n".join(
            (
                f"Title: {title}",
                f"Conversation ID: {conversation_id}",
                f"Version: {version if version is not None else 'unknown'}",
                f"Messages: {self.state.message_total if loaded else 'unknown'}",
                f"Workspace: {workspace}",
                f"Updated: {updated}",
                f"Keywords: {keywords or 'unknown'}",
                "Authority: local saved conversation",
            )
        )

    def _status_text(self) -> str:
        state = self.state
        list_status = str(self.loaded_metadata.get("_list_status") or "").strip()
        if list_status:
            return list_status
        list_summary = str(self.loaded_metadata.get("_list_summary") or "").strip()

        selected_title = str(
            self.selected_metadata.get("title")
            or state.selected_id
            or "selected conversation"
        )
        loaded_title = str(
            self.loaded_metadata.get("title")
            or state.loaded_id
            or "loaded conversation"
        )

        def with_list_summary(copy: str) -> str:
            if state.find_query:
                find_copy = (
                    f"Find: {len(state.find_matches)} exact "
                    f"{'match' if len(state.find_matches) == 1 else 'matches'}."
                    if state.find_complete
                    else "Searching complete transcript…"
                )
                copy = f"{copy} · {find_copy}"
            return f"{list_summary} · {copy}" if list_summary else copy

        if state.bulk_active:
            if state.bulk_loaded_preview_selected is True:
                preview = "The retained transcript is included and remains read-only."
            elif state.bulk_loaded_preview_selected is False:
                preview = (
                    "The retained transcript is not included and remains read-only."
                )
            else:
                preview = "No transcript is retained; Read and Info remain available."
            return with_list_summary(
                f"Bulk selection: {state.bulk_selected_count} conversations. {preview}"
            )
        if state.unavailable:
            copy = state.error or "Conversation unavailable."
            if state.selected_id:
                copy = f"{copy} Selected {selected_title} ({state.selected_id})."
            if state.loaded_id and state.loaded_id != state.selected_id:
                copy += f" Showing {loaded_title} ({state.loaded_id})."
            return with_list_summary(copy)
        if state.error:
            if state.loaded_id:
                copy = state.error
                if state.selected_id:
                    copy += f" Selected {selected_title} ({state.selected_id})."
                copy += f" Showing {loaded_title} ({state.loaded_id})."
                return with_list_summary(copy)
            selected = state.selected_id
            if selected:
                copy = f"{state.error} Selected {selected_title} ({selected})."
                if state.loaded_id and state.loaded_id != selected:
                    copy += f" Showing {loaded_title} ({state.loaded_id})."
                return with_list_summary(copy)
            return with_list_summary(state.error)
        if state.loading:
            selected = state.selected_id or "selected conversation"
            if state.loaded_id and state.loaded_id != selected:
                return with_list_summary(
                    f"Loading {selected_title} ({selected}); showing "
                    f"{loaded_title} ({state.loaded_id}) until ready."
                )
            return with_list_summary(f"Loading {selected_title} ({selected})…")
        if state.loaded_id:
            suffix = "complete" if state.complete else "loading more"
            # task-32067: by TITLE. This line is the reader's only identity
            # cue and it read "Loaded bf20fab2-0474-…" -- a raw UUID, which
            # names nothing the user has ever seen. Untitled conversations
            # get the neutral phrase, never the id.
            name = str(self.loaded_metadata.get("title") or "").strip() or (
                "this conversation"
            )
            return with_list_summary(
                f"Loaded {name} · {len(state.messages)} of "
                f"{state.message_total} messages · {suffix}."
            )
        return with_list_summary("Select a conversation to read it here.")

    def sync_state(
        self,
        state: ConversationReaderState,
        *,
        metadata: Mapping[str, Any] | None = None,
        loaded_metadata: Mapping[str, Any] | None = None,
        selected_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Patch state, labels, and progressive message rows in place."""
        # task-32361 (critique #10, B D11): the adaptive layout asks
        # ``reader_has_item`` -- ``selected_id is not None`` -- but it is only
        # resolved from ``AdaptiveReaderShellResized``, and the Conversations
        # selection settles one refresh AFTER the shell mounts. Measured at
        # 235x52 with a conversation open: list 137, Reader 44, because the
        # empty-Reader rule (task-31979) had already given the list the
        # Reader's columns and nothing re-asked. Announce the flip from the
        # one place every selection change reaches this pane; the screen's
        # existing ``@on(AdaptiveReaderShellResized)`` re-resolves.
        had_item = self.state.selected_id is not None
        self.state = state
        if self.is_mounted and (state.selected_id is not None) != had_item:
            self.post_message(AdaptiveReaderShellResized())
        if loaded_metadata is not None or metadata is not None:
            self.loaded_metadata = dict(loaded_metadata or metadata or {})
        if selected_metadata is not None:
            self.selected_metadata = dict(selected_metadata)
        if not self.is_mounted:
            return

        try:
            read = self.query_one("#library-conversation-reader-read", Button)
            info = self.query_one("#library-conversation-reader-info", Button)
            status = self.query_one("#library-conversation-reader-status", Static)
            find = self.query_one("#library-conversation-reader-find", Input)
            messages = self.query_one(
                "#library-conversation-reader-messages", VerticalScroll
            )
            info_body = self.query_one("#library-conversation-reader-info-body", Static)
            open_console = self.query_one("#library-conversation-open-console", Button)
            blocked_reason = self.query_one(
                "#library-conversation-open-console-blocked", Static
            )
            link = self.query_one("#library-conversation-link-workspace", Button)
            receipt = self.query_one("#library-conversation-link-receipt", Static)
            undo = self.query_one("#library-conversation-link-undo", Button)
            retry = self.query_one("#library-conversation-reader-retry", Button)
            find_position = self.query_one(
                "#library-conversation-find-position", Static
            )
            find_navigation = self.query_one("#library-conversation-find-navigation")
            find_buttons = tuple(
                self.query_one(f"#library-conversation-find-{direction}", Button)
                for direction in ("previous", "next")
            )
            source = self.query_one("#library-conversation-use-source", Button)
            archive_buttons = {
                action: self.query_one(f"#library-conversation-{action}", Button)
                for action in ("archive", "restore")
            }
        except (NoMatches, QueryError):
            # A retained-reader recompose can briefly leave the mounted
            # parent with an incomplete child tree. State and metadata were
            # assigned above, so the replacement compose will render this
            # arrival without a worker exception.
            return
        read.set_class(state.mode == "read", "-selected")
        info.set_class(state.mode == "info", "-selected")
        status.update(self._status_text())
        if find.value != state.find_query and not find.has_focus:
            find.value = state.find_query
        find.display = state.mode == "read"
        find_key = (state.loaded_id, state.loaded_generation, state.find_query)
        if find_key != self._find_navigation_key:
            self._find_navigation_key = find_key
            self._find_navigation_index = -1
            self._pending_find_navigation_index = None
            find_position.update("")
        find_navigation.display = state.mode == "read"
        for button in find_buttons:
            button.disabled = not (
                state.loaded_actions_eligible
                and state.find_complete
                and state.find_matches
            )
        messages.display = state.mode == "read"
        info_body.update(self._metadata_text())
        info_body.display = state.mode == "info"

        open_console.label = (
            "Restore and resume"
            if (
                self.loaded_metadata.get("archived")
                or self.loaded_metadata.get("workspace_archived")
            )
            else "Resume conversation"
        )
        open_console.disabled = not state.loaded_actions_eligible
        open_console.tooltip = _open_console_disabled_tooltip(state)
        source.label = self._source_label()
        source.disabled = not self._source_press_enabled()
        source.tooltip = self._open_console_tooltip()
        for action, button in archive_buttons.items():
            button.display = bool(self.loaded_metadata.get("archived")) == (
                action == "restore"
            )
            button.disabled = not state.loaded_actions_eligible
        blocked_reason.update(self._blocked_reason_line())
        blocked_reason.display = bool(self._blocked_reason_line())
        link.display = self._workspace_link_offered()
        receipt_workspace = self._workspace_link_receipt()
        receipt.update(
            f"✓ linked · {receipt_workspace} · this conversation can now "
            "be used in Console"
        )
        receipt.display = bool(receipt_workspace)
        undo.display = bool(receipt_workspace)
        retry.display = bool(state.error or state.unavailable)

        self._message_sync_generation += 1
        self.call_later(self._sync_messages, self._message_sync_generation)

    @on(Button.Pressed, "#library-conversation-find-previous")
    @on(Button.Pressed, "#library-conversation-find-next")
    def move_find_match(self, event: Button.Pressed) -> None:
        """Cycle matches in the exact loaded transcript without changing sessions.

        Args:
            event: Previous/next press consumed by this Reader.
        """
        event.stop()
        matches = self.state.find_matches
        if (
            not self.state.loaded_actions_eligible
            or not self.state.find_complete
            or not matches
        ):
            return
        direction = -1 if (event.button.id or "").endswith("previous") else 1
        previous = (
            self._pending_find_navigation_index
            if self._pending_find_navigation_index is not None
            else self._find_navigation_index
        )
        self._pending_find_navigation_index = (
            (len(matches) - 1 if direction < 0 else 0)
            if previous < 0
            else (previous + direction) % len(matches)
        )
        self._finish_find_navigation()

    def _finish_find_navigation(self) -> None:
        """Publish a position only after its current transcript row is revealed."""
        index = self._pending_find_navigation_index
        if index is None or not self.state.loaded_actions_eligible:
            return
        matches = self.state.find_matches
        if not self.state.find_complete or index >= len(matches):
            return
        match = matches[index]
        if not self.focus_find_match(match.message_id):
            return
        self._pending_find_navigation_index = None
        self._find_navigation_index = index
        self.query_one("#library-conversation-find-position", Static).update(
            f"Match {index + 1} of {len(matches)} · message {match.message_index + 1}"
        )

    async def _sync_messages(self, generation: int) -> None:
        """Mount or patch stable message rows after the current compose settles."""
        if generation != self._message_sync_generation or not self.is_mounted:
            return
        try:
            container = self.query_one(
                "#library-conversation-reader-messages", VerticalScroll
            )
        except (NoMatches, QueryError):
            return
        mounted = {
            str(getattr(row, "message_id", "")): row
            for row in container.children
            if getattr(row, "message_id", None)
        }
        desired_ids = {message.message_id for message in self.state.messages}
        stale = [
            row for message_id, row in mounted.items() if message_id not in desired_ids
        ]
        if stale:
            await container.remove_children(stale)
        for message in self.state.messages:
            row = mounted.get(message.message_id)
            if row is None or not row.is_mounted:
                await container.mount(self._message_widget(message))
            else:
                row.update(self._message_copy(message))
        if generation == self._message_sync_generation and self.is_mounted:
            self._finish_find_navigation()
            self.post_message(
                self.MessagesSynced(self.state.generation, self.state.find_query)
            )

    def focus_find_match(self, message_id: str) -> bool:
        """Focus and reveal one stable message reference."""
        for row in self.query(".library-conversation-reader-message"):
            if getattr(row, "message_id", None) == message_id:
                row.scroll_visible(animate=False)
                row.focus(scroll_visible=False)
                return True
        return False
