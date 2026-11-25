"""Dispatch Research links only after the destination takes navigation ownership."""

from ...Research_Workspace.owner_links import (
    RESEARCH_OWNER_LINKS,
    can_open_workspace_in_console,
)
from ..Console_Modules.research_workspace_handoff import queue_workspace_handoff
from ..Navigation.main_navigation import NavigateToScreen


def navigate_to_research_owner(screen, action, ref) -> None:
    link = RESEARCH_OWNER_LINKS[action]
    if action not in {"console", "agent-task"}:
        screen.post_message(NavigateToScreen(link.route, dict(link.context)))
        return
    if not can_open_workspace_in_console(ref):
        screen.notify(
            "Choose a Local workspace to open its context in Console.",
            severity="warning",
        )
        return
    capture = screen.controller.capture_request()
    if capture.ref != ref:
        return
    app = screen.app

    def settled(ok):
        if not ok or not screen.controller.is_current_request(capture):
            return
        target = app.screen

        def open_context():
            from ..Screens.chat_screen import ChatScreen

            if app.screen is target and isinstance(target, ChatScreen):
                queue_workspace_handoff(target, ref, create_chat=action == "agent-task")

        app.call_after_refresh(open_context)

    screen.post_message(NavigateToScreen(link.route, on_completion=settled))
