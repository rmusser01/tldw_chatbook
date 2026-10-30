"""Console code-review report presentation (ADR-281).

`ChatScreen` keeps only the binding Textual resolves by name
(``open_code_review_reports``, the ``u`` key) and one attach call
(``attach_app``); the policy lives here. Presenting needs the
review modal, a presentation-layer widget, so the Chat-layer controller
reaches it only through the presenter this module installs on attach.
"""

from __future__ import annotations

import asyncio
from typing import Any

_REPRESENTING: set[asyncio.Task] = set()


def attach_app(controller: Any, app: Any) -> None:
    """Give the controller the app, the report presenter, and pending originals.

    Args:
        controller: The Console chat controller a view is attaching to.
        app: The application object, which outlives every view.
    """
    controller.app = app
    controller.review_report_presenter = present_code_review_original
    coordinator = getattr(controller, "_code_review", None)
    if coordinator is None or not coordinator._reviews:
        return  # never build the review owner just to find nothing pending
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    # Presenting hops back onto this loop, so it must run off it.
    task = loop.create_task(asyncio.to_thread(coordinator.present_pending))
    _REPRESENTING.add(task)
    task.add_done_callback(_REPRESENTING.discard)


def present_code_review_original(
    controller: Any, session_id, review_id, report
) -> bool:
    """Acknowledge mounted native originals before an authorized verifier starts."""
    from ...Chat.console_chat_models import ConsoleMessageRole
    from ...Widgets.Console.console_code_review_modal import ConsoleCodeReviewModal

    def available():
        cancel = controller._active_cancel_events.get(session_id)
        return (
            not controller._disposed
            and session_id not in controller._session_close_generations
            and controller.store.active_session_id == session_id
            and not (cancel is not None and cancel.is_set())
        )

    async def present():
        if not available():
            return False
        try:
            _, service = controller.code_review.report_service(session_id, review_id)
            page = await asyncio.to_thread(
                ConsoleCodeReviewModal.literal_page, report["original_page"]
            )
            modal = controller.app.screen
            if (
                not isinstance(modal, ConsoleCodeReviewModal)
                or modal._review_id != review_id
            ):
                modal = ConsoleCodeReviewModal(service=service, review_id=review_id)
                await controller.app.push_screen(modal)
            if not available():
                await modal.request_safe_cancel(source="review_closed")
                return False
            if not modal.present_original(report["slot_id"], page):
                return False
            controller.store.append_message(
                session_id,
                role=ConsoleMessageRole.ASSISTANT,
                content=f"Original report {report['slot_id']} for review {review_id} is open in Review reports (u). Full originals and raw output are retained for this session; export to keep them. Saved chat references cannot restore an expired report bundle.",
            )
            return True
        except Exception:  # noqa: BLE001 - closed UI never acknowledges
            # A closed/unavailable screen does not acknowledge presentation.
            return False

    loop = controller._fleet_wake._loop
    try:
        on_loop = asyncio.get_running_loop() is loop
    except RuntimeError:
        on_loop = False
    if not on_loop and loop is not None and loop.is_running():
        call_from_thread = getattr(controller.app, "call_from_thread", None)
        if callable(call_from_thread):
            return bool(call_from_thread(present))
    return False


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
