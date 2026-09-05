"""Generic full-screen host for one pushed workbench-pane widget instance.

redesign PR-4, Task 1 (spec §11's narrow-layout push pattern): rather than
reparenting an already-mounted pane widget out of the Schedules workbench
into an overlay, a caller hands over a ``widget_factory`` that builds a
FRESH instance on every push. Two pushes -- sequential, or in principle
started from two different callers -- therefore never share state through
a widget still attached to the pane behind. No existing precedent for
"push the same widget class as a fresh full-screen instance" existed in
this codebase before this module (survey §2); the closest prior art was
the Queue/Automations tabs' own sibling ``DefinitionDetail`` instances
(the Automations one retired in task 5), which already established that
two independent instances of the same widget class, each self-syncing
with no shared state, is architecturally normal here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Header


class WorkbenchHostScreen(Screen):
    """Pushes ``widget_factory()``'s fresh widget instance full-screen.

    ``Esc`` pops. An optional ``dismissed`` hook runs once on pop, letting
    the pane behind refresh itself (e.g. re-read data that may have
    changed while the overlay was open) without either screen needing to
    know about the other's internals -- the staleness re-home consumer
    for later tasks, and this task's own conflicts-badge repoint.
    """

    # TASK-24459: the widgets this host receives are scheduling panes whose
    # `scheduling-*` rules live on the app-loaded scheduling sheet
    # (`TldwCli._ensure_screen_owned_css`). The host is only ever pushed
    # from `SchedulesWorkbench`, after the navigation that loaded the sheet
    # -- deliberately no `CSS_PATH` here, for the same harness-tier reason
    # documented on `SchedulesWorkbench`.

    BINDINGS: ClassVar = [Binding("escape", "dismiss_screen", "Back")]

    def __init__(
        self,
        widget_factory: Callable[[], Widget],
        *,
        title: str,
        dismissed: Callable[[], None] | None = None,
        route_message: Callable[[Message], None] | None = None,
    ) -> None:
        """Initialize the host.

        Args:
            widget_factory: Builds one fresh widget instance to host.
                Called exactly once, from ``compose()`` -- never reused
                across pushes; the caller is responsible for passing a
                factory that itself builds a new instance per call.
            title: Shown in the screen's ``Header``.
            dismissed: Optional hook run once when this screen pops
                (``Esc``), before the pop completes.
            route_message: Optional hook called with EVERY message this
                screen processes (task 6). A hosted pane that posts
                request messages for a handler living on the screen
                BEHIND needs them relayed: Textual bubbles a message
                widget -> parent -> Screen -> App, and the pane behind is
                on a different screen, so those messages would otherwise
                die here. The hook is deliberately unfiltered -- which
                messages are worth relaying is the caller's business, not
                a generic host's (`SchedulesWorkbench._route_pushed_
                detail_message` allowlists its own).
        """
        super().__init__()
        self._widget_factory = widget_factory
        self.title = title
        self._dismissed = dismissed
        self._route_message = route_message

    def compose(self) -> ComposeResult:
        """Header (title) + the hosted widget (fills remaining space) + Footer (Esc hint)."""
        yield Header()
        body = self._widget_factory()
        body.add_class("workbench-host-body")
        yield body
        yield Footer()

    async def _on_message(self, message: Message) -> None:
        """Dispatch normally, then hand the message to ``route_message``.

        `MessagePump._on_message` is the only seam that sees a message
        AFTER this screen's own handlers and its bubble to `App`; Textual
        exposes no public "unhandled message" hook on `Screen` (verified
        against `textual/message_pump.py` -- neither `Screen` nor
        `Widget` overrides it, so the `super()` call is the whole of the
        normal path).
        """
        await super()._on_message(message)
        if self._route_message is not None:
            self._route_message(message)

    def action_dismiss_screen(self) -> None:
        """``Esc``: run the on-dismiss hook (if any), then pop.

        Mirrors ``Screen.dismiss()``'s own ordering (result callback before
        the pop) and its documented safe-usage rule: call without
        awaiting from within an action handler.
        """
        if self._dismissed is not None:
            self._dismissed()
        self.dismiss()
