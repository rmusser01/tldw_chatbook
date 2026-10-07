# confirmation_dialog.py
# Description: Modal confirmation dialog for unsaved changes and other confirmations
#
# Imports
import asyncio
import inspect
from typing import Any, Optional, Callable

#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Label, Static

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
#
#######################################################################################################################
#
# Classes:


class ConfirmationDialog(SafeModalDismissMixin, ModalScreen[bool]):
    """
    A modal confirmation dialog for user actions.

    This dialog displays a message and provides confirm/cancel options.
    """

    # Escape mirrors the Cancel button so keyboard users can dismiss the
    # dialog without reaching for the mouse. Dismissing is always the safe
    # (non-destructive) outcome: confirm stays click/enter-on-button only.
    BINDINGS = [
        Binding("escape", "request_safe_cancel", "Cancel", show=False),
    ]
    SAFE_MODAL_CONTENT = "#confirmation-dialog"
    AUTO_FOCUS = "#cancel-button"

    # CSS for styling
    DEFAULT_CSS = """
    ConfirmationDialog {
        align: center middle;
    }
    
    ConfirmationDialog > Container,
    ConfirmationDialog > .confirmation-scroll {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    /* (.dialog-title rule removed, ADR-161 task 4: the app bundle's
       canonical .dialog-title in components/_dialogs.tcss supplies the
       centered/bold/margin styling -- app CSS outranks this DEFAULT_CSS --
       and the removed copy's lone extra declaration, width: 100%, is
       Static's default box model in a container, measured fill either way.) */

    ConfirmationDialog .dialog-message {
        margin-bottom: 2;
        width: 100%;
    }
    
    ConfirmationDialog .button-container {
        align: center middle;
        height: auto;
        width: 100%;
    }
    
    ConfirmationDialog Button {
        margin: 0 1;
        min-width: 12;
    }
    
    ConfirmationDialog .confirm-button {
        background: $error;
    }
    
    ConfirmationDialog .cancel-button {
        background: $primary;
    }
    """

    def __init__(
        self,
        title: str = "Confirm Action",
        message: str = "Are you sure you want to proceed?",
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
        confirm_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize the confirmation dialog.

        Args:
            title: Dialog title
            message: Message to display
            confirm_label: Label for confirm button
            cancel_label: Label for cancel button
            confirm_callback: Callback when confirmed
            cancel_callback: Callback when cancelled
        """
        super().__init__(**kwargs)
        self.title = title
        self.message = message
        self.confirm_label = confirm_label
        self.cancel_label = cancel_label
        self.confirm_callback = confirm_callback
        self.cancel_callback = cancel_callback
        self.result: bool | None = None

    def compose(self) -> ComposeResult:
        """Compose the dialog UI."""
        # TASK-32802.2: both are prose with user data interpolated into them
        # -- a conversation, watchlist, preset or file name -- never markup.
        # Rendered with markup ON, a title like `[TODO] Q3 plan` named the
        # wrong subject (` Q3 plan`), `[IMPORTANT]` named an empty one, and
        # `[/b]` raised MarkupError inside compose, so an irreversible
        # "Delete stored Full captures" confirmation never appeared at all.
        # Callers therefore must NOT pre-escape; the ones that did were
        # changed in the same commit.
        with VerticalScroll(id="confirmation-dialog", classes="confirmation-scroll"):
            yield Static(self.title, classes="dialog-title", markup=False)
            yield Label(self.message, classes="dialog-message", markup=False)

            with Horizontal(classes="button-container"):
                yield Button(
                    self.cancel_label,
                    id="cancel-button",
                    classes="cancel-button",
                    variant="primary",
                )
                yield Button(
                    self.confirm_label,
                    id="confirm-button",
                    classes="confirm-button",
                    variant="error",
                )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        event.stop()
        if event.button.id == "confirm-button":
            self.result = True
            if self.confirm_callback:
                await self.confirm_callback()
            self.dismiss(True)
        elif event.button.id == "cancel-button":
            await self.action_cancel_dialog()

    async def action_cancel_dialog(self) -> None:
        """Cancel the dialog (Escape and the Cancel button share this path)."""
        await self.request_safe_cancel(source="button")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Run the existing callback once, then return the exact cancel value."""
        del source
        self.result = False
        if self.cancel_callback:
            await self.run_cancel_effect_once(self.cancel_callback)
        self.dismiss_safe_once(False)


# --- The quit flow's screens and prompts (TASK-33622.10) ---------------------
#
# These live here, not in ``app_lifecycle.py`` (which runs the flow), because
# that module is under a size ratchet and this one is already resident at
# boot -- a new module would grow the ADR-097 boot census instead.


def quit_confirmation_screens(app: Any) -> list[Any]:
    """Return the screens whose ``confirm_quit``/``prepare_for_quit`` run, top first.

    Ctrl+Q is a priority binding, so the quit flow can start while a modal is
    on top. ``app.screen`` is then the modal, and the screen holding the
    unsaved work (Settings' theme editor, the Chunking Lab) sits beneath it.
    Asking only ``app.screen`` would quit straight past that screen's own
    prompt, so walk down through every modal to the first non-modal screen --
    the one the user is actually in. Screens below that one are not
    consulted, exactly as when no modal is open.

    Args:
        app: The running app (or a quit-flow harness exposing ``screen``).

    Returns:
        The active screen first, then -- only when it is modal -- each screen
        beneath it down to and including the topmost non-modal screen.
    """
    current = app.screen
    screens = [current]
    if not getattr(current, "is_modal", False):
        return screens
    stack = list(app.screen_stack)
    if current not in stack:
        return screens
    for screen in reversed(stack[: stack.index(current)]):
        screens.append(screen)
        if not getattr(screen, "is_modal", False):
            break
    return screens


async def confirm_quit_screens(screens: list[Any]) -> bool:
    """Ask each screen's ``confirm_quit``, top first; stop at the first refusal.

    Args:
        screens: From ``quit_confirmation_screens``.

    Returns:
        False when a screen answered False (stay); True otherwise.
    """
    for screen in screens:
        confirm_quit = getattr(screen, "confirm_quit", None)
        if not callable(confirm_quit):
            continue
        decision = confirm_quit()
        if inspect.isawaitable(decision):
            decision = await decision
        if decision is False:
            return False
    return True


async def prepare_quit_screens(screens: list[Any]) -> None:
    """Run each screen's ``prepare_for_quit``, top first, once the quit is approved.

    Args:
        screens: The same list ``confirm_quit_screens`` was asked with.
    """
    for screen in screens:
        prepare_for_quit = getattr(screen, "prepare_for_quit", None)
        if not callable(prepare_for_quit):
            continue
        preparation = prepare_for_quit()
        if inspect.isawaitable(preparation):
            await preparation


# Ctrl+Q is a priority binding, so the quit flow can push its prompts while a
# modal is open. In Textual 8.2.8 ``Screen.dismiss()`` resolves its own result
# and then calls ``app.pop_screen()``, which pops whatever is on TOP; and
# ``App.pop_screen`` drops the popped screen's result callback unresolved. So a
# covered modal that closes itself from a timer or worker pops the quit prompt
# above it, and a bare ``push_screen_wait`` on that prompt never returns: the
# quit worker hangs, ``_quit_in_progress`` stays set, and Ctrl+Q is dead for
# the rest of the session. Every prompt the quit flow owns awaits
# ``await_quit_prompt`` instead, so that cannot happen.

#: How often an open quit prompt is checked for having left the screen stack
#: unanswered: one membership test on a few-screen list, only while it is open.
PROMPT_WATCH_INTERVAL_SECONDS = 0.1
#: How long an answer may still land after its prompt has left the stack.
PROMPT_ANSWER_GRACE_SECONDS = 0.1
#: What the user is told when a quit prompt vanished before they answered it.
QUIT_CANCELLED_NOTICE = "Quit cancelled."


async def await_quit_prompt(
    app: Any,
    prompt: Screen,
    *,
    no_answer: Any,
    vanished_notice: str | None = QUIT_CANCELLED_NOTICE,
) -> Any:
    """Push ``prompt`` and return its answer, or ``no_answer`` if it vanishes.

    The quit flow's one choke point for awaiting a prompt (TASK-33622.10).
    Unlike ``push_screen_wait`` it cannot hang on a prompt that something
    else popped (see the section comment above).

    Why it polls: Textual 8.2.8 has no reliable removal signal for an
    arbitrary screen. ``Unmount`` and ``ScreenSuspend`` reach only the
    prompt's own handlers (a hook on every prompt class), and ``ScreenSuspend``
    also fires whenever another screen is pushed over it.
    ``App.screen_change_signal`` is public and synchronous, but subscriptions
    are keyed per DOM node and ``unsubscribe(node)`` drops all of that node's
    callbacks (the app and ``ConsoleSessionSwitcherModal`` both subscribe
    themselves), while the prompt itself cannot subscribe until its message
    pump starts -- after the push, leaving a window for exactly the pop that
    matters. So the answer is awaited event-driven, and only "has it left
    the stack?" is polled, cheaply, for the prompt's lifetime alone.

    ``push_screen(..., wait_for_dismiss=True)`` appends the prompt to the
    stack synchronously, so it is on the stack from the moment this call
    pushes it; leaving it later is unambiguous. ``Screen.dismiss`` resolves
    the answer BEFORE it pops, so an answered prompt never reads as
    vanished; the grace only guards that ordering. The orphaned answer
    future is deliberately left alone: cancelling it would make a late
    ``dismiss`` raise inside the prompt's own handler.

    When the prompt vanished because a covered screen dismissed itself,
    Textual has delivered that screen's result but popped the prompt in its
    place, leaving it on the stack as a zombie whose next close raises
    ``InvalidStateError`` and exits the app. ``SafeModalDismissMixin``
    modals refuse such a dismiss, so this only meets plain ones; for those
    the close the screen asked for is finished here.

    Unlike ``push_screen_wait`` it does not first await the app's pending
    ``call_next`` callbacks. Deliberately: at quit time those are earlier
    pops' own removals, which a push does not conflict with, and the prompt
    still lands on the stack synchronously.

    Must run inside a worker, as ``push_screen_wait`` must.

    Args:
        app: The running app.
        prompt: The modal to push; its dismiss result is the answer.
        no_answer: Returned when the prompt left the stack unanswered. Every
            caller passes its own Stay / Keep editing value.
        vanished_notice: Toast shown when that happens; ``None`` for none.

    Returns:
        The prompt's answer, or ``no_answer`` if it vanished unanswered.
    """
    answer = app.push_screen(prompt, wait_for_dismiss=True)
    while True:
        done, _ = await asyncio.wait({answer}, timeout=PROMPT_WATCH_INTERVAL_SECONDS)
        if done:
            return answer.result()
        if prompt in app.screen_stack:
            continue
        done, _ = await asyncio.wait({answer}, timeout=PROMPT_ANSWER_GRACE_SECONDS)
        if done:
            return answer.result()
        finished = _finish_interrupted_closes(app)
        logger.warning(
            "{} left the screen stack unanswered; treating it as Stay "
            "(finished {} interrupted close(s))",
            type(prompt).__name__,
            finished,
        )
        if vanished_notice:
            app.notify(vanished_notice, severity="warning")
        return no_answer


def _closed_but_still_stacked(screen: Any) -> bool:
    """Whether ``screen``'s own dismiss already delivered a result.

    Textual 8.2.8 gives every push a ``ResultCallback`` whose future is
    resolved only by that screen's ``dismiss``. Resolved -- not cancelled,
    which a cancelled ``push_screen_wait`` waiter does to a screen still
    legitimately open -- while the screen is still on the stack means its
    pop went to the screen that was above it.
    """
    callbacks = getattr(screen, "_result_callbacks", None)
    if not callbacks:
        return False
    future = getattr(callbacks[-1], "future", None)
    return future is not None and future.done() and not future.cancelled()


def _finish_interrupted_closes(app: Any) -> int:
    """Pop each top screen whose own close went astray; return how many."""
    finished = 0
    while len(app.screen_stack) > 1 and _closed_but_still_stacked(app.screen):
        app.pop_screen()
        finished += 1
    return finished


async def confirm_quit_discarding_edits(
    screen: Screen,
    message: str,
    *,
    title: str = "Discard changes and quit?",
    confirm_label: str = "Discard and quit",
    cancel_label: str = "Keep editing",
) -> bool:
    """Ask whether quitting may discard a modal's unsaved edits (TASK-33622.10).

    Ctrl+Q is a priority binding, so the app's quit flow can start while a
    modal is open, and it asks that modal's ``confirm_quit`` first. A modal
    that guards its own close with a discard prompt answers ``confirm_quit``
    with this, so quitting asks the same question instead of dropping the
    edits silently.

    The default words fit unsaved edits. A modal whose close guard protects
    something else -- a generated video, an interview held only in memory, a
    side effect that quitting keeps -- passes its own, so the prompt never
    claims a discard that is not one.

    It waits on the pushed dialog through ``await_quit_prompt``, so it must
    run inside a worker; the app's quit flow is one.

    Args:
        screen: The modal holding the edits.
        message: What would be lost, in that modal's own words.
        title: The prompt's title.
        confirm_label: The button that lets the quit proceed.
        cancel_label: The button (and Escape) that stays in the modal.

    Returns:
        True when the user chose to quit; False to stay, including when the
        prompt vanished before it was answered.
    """
    choice = await await_quit_prompt(
        screen.app,
        ConfirmationDialog(
            title=title,
            message=message,
            confirm_label=confirm_label,
            cancel_label=cancel_label,
        ),
        no_answer=False,
    )
    return choice is True


class UnsavedChangesDialog(ConfirmationDialog):
    """
    Specialized confirmation dialog for unsaved changes.
    """

    def __init__(
        self,
        tab_title: str = "Untitled",
        confirm_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize unsaved changes dialog.

        Args:
            tab_title: Title of the tab with unsaved changes
            confirm_callback: Callback when user confirms close
            cancel_callback: Callback when user cancels
        """
        super().__init__(
            title="Unsaved Changes",
            message=f'The tab "{tab_title}" has unsaved changes.\n\nAre you sure you want to close it?',
            confirm_label="Close Without Saving",
            cancel_label="Keep Open",
            confirm_callback=confirm_callback,
            cancel_callback=cancel_callback,
            **kwargs,
        )


#
# End of confirmation_dialog.py
#######################################################################################################################
