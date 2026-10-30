"""Console ``u`` key: reopen retained code-review reports (ADR-281).

`ChatScreen` keeps only the binding Textual resolves by name
(``open_code_review_reports``); the policy lives here.
"""

from __future__ import annotations

from typing import Any


def open_code_review_reports(screen: Any) -> None:
    """Open retained review evidence for the current Console chat.

    Args:
        screen: The Console screen; inert while first-run setup blocks it.
    """
    if screen._console_setup_modal_blocking():
        return
    from ...Agents.code_review_models import ReviewRefused
    from ...Widgets.Console.console_code_review_modal import ConsoleCodeReviewModal

    controller = screen._ensure_console_chat_controller()
    session_id = controller.store.active_session_id
    if not session_id:
        screen.notify("No active conversation.", severity="warning")
        return
    try:
        review_id, service = controller.code_review.report_service(session_id)
    except ReviewRefused:
        screen.notify("No retained review reports in this session.", severity="warning")
        return
    screen.app.push_screen(ConsoleCodeReviewModal(service=service, review_id=review_id))
