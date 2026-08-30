"""Floating selection-action menu (console selection phase 1).

Mounted on the owning SCREEN at an absolute offset (the same anchoring
mechanism Textual's own tooltips use); NOT a ModalScreen (modals are
layer-centered and cannot anchor at a cell) and NOT a docked transcript
child. Live-spike round 3: the previous ``dock: top`` + ``styles.offset``
combination painted the menu translated by the offset while clipping it to
the un-translated dock slot -- the user saw one button, and hit-tests used
the un-translated region so the other buttons were unclickable. Mounting on
the screen with ``absolute_offset`` folds the position into the widget's
region, so paint, clipping, and hit-testing all agree. Post-layout the
menu clamps itself inside the OWNING TRANSCRIPT's visible box -- never the
bare screen -- because the composer + status bar live below the transcript
(a screen-clamped bottom release painted the menu over them; live spike
2026-08-16).

Escape and click-outside dismiss with no side effects (task-16211 modal
contract, recorded by ADR-068). Keyboard navigation: up/down cycles the
action buttons (skipping disabled ones), Enter activates the focused one,
Escape closes.

Phase 3 task 2: when ``feedback_available`` (selection in agent output)
the menu also offers ``Request changes | LGTM | Comment``; without an
active run the first two render disabled with a dim hint line (Comment
stays reachable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from weakref import WeakSet

from textual import on
from textual.binding import Binding
from textual.containers import Vertical
from textual.dom import NoScreen
from textual.events import Click, Key, Resize
from textual.geometry import Offset, Region
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

if TYPE_CHECKING:  # pragma: no cover - typing only
    from textual.screen import Screen

#: Every constructed, not-yet-collected selection menu (TASK-21119).
#:
#: Registration happens in ``__init__`` -- synchronously, and strictly BEFORE
#: the widget can be attached to the DOM -- so the registry can never MISS a
#: mounted menu. That direction is the load-bearing one: an under-reporting
#: registry would silently stop dismissal. It CAN over-report (a menu built
#: but never mounted, or removed before its ``Unmount`` ran), which is why
#: liveness is never read from the registry itself: ``selection_menus_on_
#: screen`` re-derives it from the DOM (``menu.screen``), and over-reporting
#: only costs the walk the fix was avoiding. Weak references keep abandoned
#: menus from pinning widget trees in memory.
_LIVE_SELECTION_MENUS: "WeakSet[ConsoleSelectionMenu]" = WeakSet()


def selection_menus_on_screen(screen: "Screen[object]") -> list["ConsoleSelectionMenu"]:
    """Selection menus currently attached under ``screen``.

    The O(1)-ish replacement for ``screen.query(ConsoleSelectionMenu)`` (a
    full-screen DOM walk) on the per-press dismissal path. At most one menu
    is ever mounted, so the candidate set is tiny; each candidate's
    attachment is confirmed against the live DOM (``menu.screen``), never
    against bookkeeping, so a stale registry entry cannot produce a phantom
    menu -- and a detached menu (``NoScreen``) drops out on its own.

    Args:
        screen: The screen whose subtree is being inspected.

    Returns:
        The attached menus, in unspecified order (callers dismiss all of
        them).
    """
    menus: list[ConsoleSelectionMenu] = []
    for menu in _LIVE_SELECTION_MENUS:
        if menu.parent is None:
            continue  # never mounted, or already detached (cheap arm)
        try:
            if menu.screen is screen:
                menus.append(menu)
        except NoScreen:
            continue  # attached to an orphaned subtree mid-teardown
    return menus


#: Shrink-guard class: added by the measured clamp when the owner box is
#: shorter than even the compact menu; drops the container border and the
#: hint line (3 rows) before the top-out tie-break. No actions are hidden.
_SHRUNK_CLASS = "shrunk-for-short-owner"

#: Shown (dim, inside the menu) and carried on the disabled buttons'
#: tooltips when Request changes / LGTM are run-gated (phase 3 task 2).
_NO_RUN_HINT = "No active run — start a run to send review feedback"


class ConsoleSelectionQuoteRequested(Message):
    """Bubbled when the user adds the active selection as a quote to the chat."""

    def __init__(self, quote: str) -> None:
        super().__init__()
        self.quote = quote


class ConsoleSideChatRequested(Message):
    """Bubbled when the user opens an ephemeral side chat about a selection.

    Console selection phase 2: the transcript posts this after the menu's
    "More Details" (``mode="more-details"``) or "Ask in Side Chat"
    (``mode="ask"``) action; the owning ``ChatScreen`` resolves config +
    gateway and pushes ``ConsoleSideChatModal``.
    """

    MODE_MORE_DETAILS = "more-details"
    MODE_ASK = "ask"

    def __init__(self, quote: str, mode: str) -> None:
        super().__init__()
        self.quote = quote
        self.mode = mode


class ConsoleSelectionNoteRequested(Message):
    """Bubbled when the user saves the active selection as a note.

    task-18156 Task 6 (maintainer request): the transcript posts this after
    the menu's "Create note" action; the owning ``ChatScreen`` derives a
    title from the quote and writes the note off-thread through the store's
    persistence DB. Available for EVERY selection -- notes are not review
    feedback, so there is no row-kind or run gating.
    """

    def __init__(self, quote: str) -> None:
        super().__init__()
        self.quote = quote


class ConsoleSelectionFeedbackRequested(Message):
    """Bubbled when the user sends review feedback about a selection.

    Console selection phase 3: the transcript posts this after the menu's
    "Request changes" / "LGTM" / "Comment" actions on a selection in agent
    output; the owning ``ChatScreen`` composes the structured feedback
    message (action header + quoted selection + optional comment) and
    routes it as the next user message via the prompt queue.
    """

    ACTION_REQUEST_CHANGES = "request-changes"
    ACTION_LGM = "lgm"
    ACTION_COMMENT = "comment"

    def __init__(
        self, action: str, quote: str, anchor_message_id: str | None = None
    ) -> None:
        super().__init__()
        self.action = action
        self.quote = quote
        # task-17169: the id of the row the selection came from. Durable
        # feedback needs an anchor the quote cannot provide -- quoted text
        # does not survive a re-render or identify WHICH message it came
        # from when the same text appears twice. None for callers that
        # have no origin row (the feedback is still dispatched; only the
        # audit record is skipped).
        self.anchor_message_id = anchor_message_id


class ConsoleSelectionMenu(Vertical):
    """Floating stacked menu anchored at the selection release cell."""

    can_focus = True

    BUNDLED_CSS = """
    ConsoleSelectionMenu {
        position: absolute;
        /* Live spike 2026-08-16: textual 8.2.8's vertical layout excludes
           position:absolute children from sibling stacking but still feeds
           their height into the fr denominator -- mounted on the screen, the
           menu shrank the 1fr #screen-content sibling by its own height and
           floated the composer above dead rows (the "black bar").
           overlay:screen is the style that removes an overlay from the
           container's flow math entirely. */
        overlay: screen;
        width: auto;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    /* One row per action (clamp-fix review): the library Button chrome
       (line-pad 1 + tall bottom border) stacked 3 rows per button, so the
       6-action feedback variant measured ~24 rows and towered over short
       transcripts, bleeding past the owner-box clamp onto the composer.
       !important is load-bearing: the library's variant/hover/disabled
       rules re-assert border-top/bottom at equal-or-higher specificity. */
    ConsoleSelectionMenu Button {
        height: 1 !important;
        min-height: 1 !important;
        border: none !important;
        border-top: none !important;
        border-bottom: none !important;
        padding: 0 1 !important;
    }
    /* ANSI color mode (increment review): the library's disabled rule
       (Button:ansi.-style-flat:disabled, specificity (0,3,1)) beats the
       generic compact rule above ((0,0,2)) -- both !important, so
       specificity decides -- and re-grew tall borders on the run-gated
       pair (2-row border-only boxes, labels clipped, 11-row menu). Per-ID
       rules ((1,0,1)) beat any class/pseudo stack textual throws. Applied
       to ALL seven action IDs, not just the two gated ones: any action may
       end up disabled, and every action must stay one row in every state
       and color mode. */
    ConsoleSelectionMenu #console-selection-add-to-chat,
    ConsoleSelectionMenu #console-selection-more-details,
    ConsoleSelectionMenu #console-selection-ask-side-chat,
    ConsoleSelectionMenu #console-selection-create-note,
    ConsoleSelectionMenu #console-selection-request-changes,
    ConsoleSelectionMenu #console-selection-lgm,
    ConsoleSelectionMenu #console-selection-comment {
        border: none !important;
        border-top: none !important;
        border-bottom: none !important;
    }
    ConsoleSelectionMenu #console-selection-feedback-hint {
        color: $text-muted;
        text-style: dim;
        width: auto;
    }
    /* Shrink guard for boxes shorter than even the compact menu: the
       measured clamp adds this class, trading the container border and
       the hint line for 3 more usable rows (last resort before the
       top-out tie-break; no actions are ever hidden). */
    ConsoleSelectionMenu.shrunk-for-short-owner {
        border: none;
    }
    ConsoleSelectionMenu.shrunk-for-short-owner #console-selection-feedback-hint {
        display: none;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "dismiss", show=False)]

    class AddToChat(Message):
        """User chose 'Add to chat' for the active selection."""

    class MoreDetails(Message):
        """User chose 'More Details' (auto-send side chat) for the selection."""

    class AskInSideChat(Message):
        """User chose 'Ask in Side Chat' (freeform) for the selection."""

    class CreateNote(Message):
        """User chose 'Create note' for the active selection."""

    class RequestChanges(Message):
        """User chose 'Request changes' review feedback for the selection.

        Phase 3 task 2: posted only when a run is active (run-gated); the
        owning transcript composes the structured feedback message.
        """

    class Lgm(Message):
        """User chose 'LGTM' review feedback for the selection (run-gated)."""

    class Comment(Message):
        """User chose 'Comment' feedback for the selection (always enabled)."""

    def __init__(
        self,
        *,
        screen_x: int,
        screen_y: int,
        has_add_to_chat: bool = True,
        owner: Widget | None = None,
        feedback_available: bool = False,
        run_active: bool = False,
        selection_top: int | None = None,
    ) -> None:
        """Anchor the menu at screen coordinates.

        Args:
            screen_x: X anchor (in cells) in SCREEN coordinates -- the
                release column of the drag.
            screen_y: Y anchor in screen coordinates; the caller passes
                ``release_row + 1`` so the menu sits just below the release
                cell.
            has_add_to_chat: Whether to offer the "Add to chat" action.
            owner: The widget that owns the selection lifecycle (the
                ``ConsoleTranscript``). The menu mounts on the SCREEN, so
                its action/dismissal messages are POSTED DIRECTLY to the
                owner instead of bubbling -- screen-level bubbling would
                never reach the transcript's handlers. ``None`` falls back
                to normal bubbling (bare test harnesses).
            feedback_available: Whether the selection sits in agent output
                (tool/diff rows), offering the review-feedback actions.
            run_active: Whether a console run is currently active; when
                False, Request changes / LGTM render disabled with the
                no-run hint (Comment stays enabled).
            selection_top: Screen y of the selected row's top, used ONLY by
                the measured clamp's bottom-overflow placement (the menu
                hops entirely above the row so its highlight strip stays
                visible); NOT part of the anchor. ``None`` (no row, or its
                region unmeasured) keeps the plain bottom-pinned clamp.
        """
        super().__init__(id="console-selection-menu")
        self._anchor = (screen_x, screen_y)
        self._owner = owner
        self._has_add_to_chat = has_add_to_chat
        self._feedback_available = feedback_available
        self._run_active = run_active
        self._selection_top = selection_top
        #: Widget holding focus before the menu grabbed it (captured in
        #: ``on_mount`` BEFORE focusing the first button); ``None`` =
        #: nothing was focused (or the capture raced teardown), so unmount
        #: falls back to the composer.
        self._previous_focus: Widget | None = None
        # TASK-21119: register BEFORE any mount can happen. Textual delivers
        # ``Mount`` asynchronously (the widget's own message pump), so an
        # ``on_mount`` registration would leave a window in which the menu is
        # already in the DOM but invisible to the screen's dismissal gate.
        _LIVE_SELECTION_MENUS.add(self)

    def compose(self):
        if self._has_add_to_chat:
            yield Button("Add to chat", id="console-selection-add-to-chat", variant="primary")
        yield Button("More Details", id="console-selection-more-details")
        yield Button("Ask in Side Chat", id="console-selection-ask-side-chat")
        yield Button("Create note", id="console-selection-create-note")
        if self._feedback_available:
            gated = not self._run_active
            request = Button(
                "Request changes",
                id="console-selection-request-changes",
                disabled=gated,
            )
            lgm = Button("LGTM", id="console-selection-lgm", disabled=gated)
            if gated:
                request.tooltip = _NO_RUN_HINT
                lgm.tooltip = _NO_RUN_HINT
            yield request
            yield lgm
            # Comment is never run-gated: it routes the same way but stays
            # reachable without an active run (phase 3: routing only).
            yield Button("Comment", id="console-selection-comment")
            if gated:
                yield Static(_NO_RUN_HINT, id="console-selection-feedback-hint")

    def on_mount(self) -> None:

        # Capture the pre-mount focus holder BEFORE focusing a menu button:
        # a drag that started from a focused transcript must return focus
        # there on dismissal, not be pulled into the composer (final review).
        try:
            self._previous_focus = self.screen.focused
        except Exception:  # noqa: BLE001 - capture is best-effort during odd teardown
            self._previous_focus = None
        self.absolute_offset = Offset(*self._anchor)
        # Only this widget knows its real extent (border + padding +
        # buttons); pull the anchor back inside the OWNING TRANSCRIPT's
        # visible box once layout has measured it, or a release near the
        # transcript's bottom edge anchors the lower buttons over the
        # composer below it (live spike 2026-08-16; pilot/OutOfBounds;
        # real terminal: unreachable and overlapping).
        self.call_after_refresh(self._clamp_within_owner)
        # Skip disabled buttons (run-gated feedback actions): focusing one
        # would drop focus entirely (disabled widgets are not focusable).
        buttons = [b for b in self.query(Button) if b.display and not b.disabled]
        if buttons:
            # scroll_visible=False: focusing must not scroll the screen to
            # the menu (it shifted the transcript out from under the
            # selection when the menu mounted).
            buttons[0].focus(scroll_visible=False)
        else:
            self.focus(scroll_visible=False)

    def on_resize(self, _event: Resize) -> None:
        """Re-clamp after late CSS measurement changes the menu's extent."""
        self.call_after_refresh(self._clamp_within_owner)

    def _clamp_within_owner(self) -> None:
        """Shift the anchor so the measured menu fits its clamp box.

        The clamp box is the OWNING TRANSCRIPT's visible region (the
        composer + status bar live below it, so the transcript box ends
        above the screen edge -- clamping to SCREEN bounds painted the menu
        over the composer on bottom-of-transcript releases, live spike
        2026-08-16). Ownerless menus (bare test harnesses) fall back to the
        screen bounds.

        Never cover the selection you just made (live spike 2026-08-16
        8:48): when the menu overflows the box bottom, pinning it to the
        box bottom lands it ON TOP of the selected row -- the reverse-video
        highlight strip (the evidence of the selection) hides behind the
        menu. If ``selection_top`` is known and the whole menu fits above
        the row, place it there instead -- preferring a one-row gap (bottom
        exactly at ``selection_top - 1``), then touching the row's top when
        the gap alone does not fit, and only then falling back to the
        bottom pin. The row top is clamped to the box so a stale sample
        can never place the menu outside the owner.
        Ordering with the shrink guard below is stable: the guard fires
        only when the menu is taller than the whole box -- in that case it
        can never fit above the selection either (``selection_top`` is at
        most the box bottom), so the measure -> shrink -> re-measure passes
        never fight this placement.
        """
        if self.parent is None or not self.is_attached:
            return
        region = self.region
        if not region:
            return
        owner = self._owner
        bounds: Region | None = None
        if owner is not None and owner.is_attached:
            owner_region = owner.region
            if owner_region:
                bounds = owner_region
        if bounds is None:
            screen_size = self.screen.size
            bounds = Region(0, 0, screen_size.width, screen_size.height)
        # Shrink guard (clamp-fix review): a box shorter than even the
        # compact menu cannot contain it at ANY offset -- trade the
        # container border + hint line for three more usable rows, then
        # re-measure in a fresh layout pass (the class check stops the
        # recursion; if it still does not fit, the top-out tie-break below
        # is the accepted last resort).
        if region.height > bounds.height and not self.has_class(_SHRUNK_CLASS):
            self.add_class(_SHRUNK_CLASS)
            self.call_after_refresh(self._clamp_within_owner)
            return
        shift_x = max(0, region.right - bounds.right)
        shift_y = max(0, region.bottom - bounds.bottom)
        if not shift_x and not shift_y:
            return
        x, y = self._anchor
        # Bottom overflow: hop entirely above the selected row when the
        # whole menu fits between the box top and the row top; the row (and
        # its highlight strip, which lives inside the row widget) stays
        # visible. Placement: exclusive bottom at ``selection_top - 1`` --
        # the menu's last occupied cell is one row clear of the row top
        # (Region.bottom is exclusive; the gap keeps the highlight strip
        # visually separated). Horizontal clamp unchanged.
        if shift_y and self._selection_top is not None:
            # Defensive bound (review follow-up): the row top is sampled
            # pre-mount; a stale or beyond-the-box sample must never pull
            # the menu outside the owner box -- clamp the effective row top
            # so the containment invariant holds unconditionally.
            selection_top = min(self._selection_top, bounds.bottom)
            above_y = selection_top - 1 - region.height
            if above_y < bounds.y:
                # No room for the one-row gap: abut the row (touching)
                # before giving up -- pinning to the box bottom instead
                # would land the menu ON the highlight this branch exists
                # to keep visible (reachable on boxes <= ~2x menu height).
                above_y = selection_top - region.height
            if above_y >= bounds.y:
                self._anchor = (
                    max(bounds.x, x - shift_x),
                    above_y,
                )
                self.absolute_offset = Offset(*self._anchor)
                return
        self._anchor = (
            max(bounds.x, x - shift_x),
            max(bounds.y, y - shift_y),
        )
        self.absolute_offset = Offset(*self._anchor)

    def on_key(self, event: Key) -> None:
        """Keyboard navigation: arrows cycle actions; Escape closes.

        Escape is handled here (not only via BINDINGS) because ancestor
        widgets -- ``ConsoleTranscript.on_key`` stops Escape for its
        clear-selection action during bubbling, before binding dispatch
        would consult this menu's BINDINGS. The focused widget's
        ``on_key`` runs first in the bubble chain (a menu button or the
        menu itself), so handling the key here is what actually fires in
        the real transcript context.
        """
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.action_dismiss()
            return
        if event.key in ("up", "down"):
            event.stop()
            event.prevent_default()
            # Skip disabled buttons (run-gated Request changes / LGTM):
            # focusing a disabled widget is a no-op that DROPS focus, which
            # would strand keyboard navigation.
            buttons = [b for b in self.query(Button) if b.display and not b.disabled]
            if not buttons:
                return
            focused = next((b for b in buttons if b.has_focus), buttons[0])
            index = buttons.index(focused)
            step = 1 if event.key == "down" else -1
            buttons[(index + step) % len(buttons)].focus()

    def _post(self, message: Message) -> None:
        (self._owner if self._owner is not None else self).post_message(message)

    @on(Button.Pressed, "#console-selection-add-to-chat")
    def _add_to_chat(self) -> None:
        self._post(self.AddToChat())

    @on(Button.Pressed, "#console-selection-more-details")
    def _more_details(self) -> None:
        self._post(self.MoreDetails())

    @on(Button.Pressed, "#console-selection-ask-side-chat")
    def _ask_side_chat(self) -> None:
        self._post(self.AskInSideChat())

    @on(Button.Pressed, "#console-selection-create-note")
    def _create_note(self) -> None:
        self._post(self.CreateNote())

    @on(Button.Pressed, "#console-selection-request-changes")
    def _request_changes(self) -> None:
        self._post(self.RequestChanges())

    @on(Button.Pressed, "#console-selection-lgm")
    def _lgm(self) -> None:
        self._post(self.Lgm())

    @on(Button.Pressed, "#console-selection-comment")
    def _comment(self) -> None:
        self._post(self.Comment())

    class Dismissed(Message):
        """Escape dismissal: the owning transcript clears the selection UI."""

    def action_dismiss(self) -> None:
        self._post(self.Dismissed())
        self.remove()

    def _on_click(self, event: Click) -> None:
        event.stop()  # clicks inside the menu must not clear anything
        # A click on the menu that did NOT land on one of its action
        # buttons (border/padding/label areas) is a popover dismissal:
        # clear the selection UI so the next click reaches the row.
        node = event.control
        while node is not None and node is not self:
            if isinstance(node, Button):
                return
            node = node.parent
        self._post(self.Dismissed())
        self.remove()

    def _on_unmount(self) -> None:
        # Best-effort registry prune (TASK-21119): dropping the entry keeps
        # the candidate set small, but correctness never depends on it --
        # ``selection_menus_on_screen`` re-checks attachment, and the weak
        # reference expires on its own if this hook is skipped in teardown.
        _LIVE_SELECTION_MENUS.discard(self)
        self._restore_previous_focus()

    def _restore_previous_focus(self) -> None:
        """Return focus to the widget that held it before the menu mounted.

        ``on_mount`` captured ``screen.focused`` before the menu grabbed
        focus; every dismissal path (escape, click-outside, add-to-chat
        cleanup) funnels through removal, so unmount is the single restore
        seam. The captured widget is restored when it is still mounted on
        the same screen -- a drag that started from a focused transcript
        must return focus there on Escape, not be pulled into the
        composer. Otherwise focus falls back to the console composer.
        Skips quietly when the fallback finds no composer (bare
        transcript/test harnesses) or when the screen is already gone
        during teardown (``self.screen`` raises NoScreen).
        """
        try:
            screen = self.screen
        except Exception:  # noqa: BLE001 - focus restore is best-effort during teardown
            return
        previous = self._previous_focus
        if previous is not None and previous is not self and previous.is_mounted:
            try:
                if previous.screen is screen:
                    previous.focus(scroll_visible=False)
                    return
            except Exception:  # noqa: BLE001, S110 - previous detached during teardown
                pass
        for composer in screen.query("#console-native-composer"):
            composer.focus(scroll_visible=False)
            return
