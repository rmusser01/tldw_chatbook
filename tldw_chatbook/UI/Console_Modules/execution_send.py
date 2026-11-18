"""Original pressed-send presentation backed by the selected command owner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import FunctionType
from uuid import uuid4

from ...Chat.console_execution_process import (
    ConsoleChannelLost,
    ConsoleExecutionClient,
    ConsoleReadinessExpired,
    ConsoleRemoteFailure,
)
from ...Chat.console_execution_protocol import _PROJECTED_INPUTS, _detach
from ...Widgets.Console.console_composer_bar import (
    ConsoleComposerBar,
    ConsoleDraftStash,
    classify_console_raw_draft,
)
from ...Widgets.Console import console_composer_bar as composer_source
from ..Screens import chat_screen as screen_source
from . import send_acknowledgement as sending
from . import sent_draft
from .execution_projection import _definition, _definition_current, _require_methods
from .execution_sessions import _require_ack
from .execution_state import _AUTHORED_INPUT_METHODS, selected_client


@dataclass(eq=False, slots=True)
class _SelectedPress:
    client: object
    screen: object
    visit: int
    source_revision: int
    session_id: str
    incarnation_id: str
    stash: ConsoleDraftStash
    authored: tuple[str, int, int]
    inputs: object
    rag_sources: tuple[str, ...]
    registration: object = None


def _current(screen, client, visit):
    if selected_client(screen) is not client or client._visit != visit:
        raise ConsoleChannelLost("Console pressed view changed")
    _require_methods(client, _AUTHORED_INPUT_METHODS)
    for name, record in _DEFINITIONS:
        if globals().get(name) is not record[0] or not _definition_current(record):
            raise RuntimeError("Console selected Send source changed")
    for module, name, record in _SOURCES:
        if getattr(module, name) is not record[0] or not _definition_current(record):
            raise RuntimeError("Console original Send helper changed")


def capture_inputs(screen, session_id, stash):
    """Freeze the real press before original ACK paint or deferred replay."""
    client = selected_client(screen)
    _current(screen, client, client._visit)
    if type(stash) is not ConsoleDraftStash:
        raise RuntimeError("Console selected Send requires its captured text")
    row = client.projection_view.session(session_id)
    if row.id != client.projection_view.active_session_id:
        raise RuntimeError("Console pressed chat changed")
    inputs = _detach(client.pressed_inputs, "pressed_inputs", 0, set())
    _PROJECTED_INPUTS(inputs, "pressed_inputs")
    press = _SelectedPress(
        client,
        screen,
        client._visit,
        client._source_revision,
        row.id,
        row.incarnation_id,
        stash,
        (stash.text, stash.generation, stash.edit_serial),
        inputs,
        screen_source._console_library_rag_source_scope(screen),
    )
    press.registration = (press, client, screen)
    return press


def request_send(screen, guard=None) -> bool:
    """Use the existing capture/defer/paint scheduler on the selected route."""
    client = selected_client(screen)
    if client is None:
        return False
    _current(screen, client, client._visit)
    composer = screen._console_composer_or_none()
    if type(composer) is not ConsoleComposerBar:
        raise RuntimeError("Console original composer is unavailable")
    _require_methods(composer, _COMPOSER_METHODS)
    stash = composer.capture_draft_for_send()
    if stash is None:
        screen.app_instance.notify("There is no text to send.", severity="warning")
        return True
    classified = classify_console_raw_draft(stash)
    if classified.kind != "chat" or classified.text.lstrip().startswith("/"):
        # These retain their separate original command owners. Never treat a
        # host command or escaped command capture as provider text by fallback.
        screen.app_instance.notify(
            "This Console command is unavailable; your draft was kept.",
            severity="warning",
        )
        return True
    if not classified.text.strip():
        return True
    session_id = screen._console_visible_send_session_id()
    if session_id != client.projection_view.active_session_id:
        screen.app_instance.notify(sending.CHAT_CHANGED_COPY, severity="warning")
        return True
    sending._schedule_visible_send_capture(screen, session_id, stash, guard)
    return True


def record_draft(screen, request) -> bool:
    """The original widget observer owns edits; scheduling never writes a mirror."""
    client = selected_client(screen)
    _current(screen, client, client._visit)
    return any(
        row.id == request.session_id for row in client.projection_view.sessions()
    )


async def dispatch_send(screen, *, session_id=None, pending_send_token=None) -> bool:
    """Join exact admission, explicit plan ACK and original custody outcomes."""
    client = selected_client(screen)
    visit = client._visit
    _current(screen, client, visit)
    pending = screen._console_pending_send
    if (
        pending_send_token is None
        or pending is None
        or pending.token is not pending_send_token
    ):
        screen.app_instance.notify(
            "Console send capture changed; your draft was kept.", severity="warning"
        )
        return False
    captured = pending.received_inputs
    screen._console_pending_send = None
    if (
        type(captured) is not _SelectedPress
        or captured.registration != (captured, client, screen)
        or captured.client is not client
        or captured.screen is not screen
        or captured.visit != visit
        or captured.source_revision != client._source_revision
        or captured.session_id != pending.session_id
        or captured.stash is not pending.stash
        or captured.authored
        != (pending.stash.text, pending.stash.generation, pending.stash.edit_serial)
    ):
        raise RuntimeError("Console original press capture changed")
    row = client.projection_view.session(captured.session_id)
    if row.incarnation_id != captured.incarnation_id:
        raise RuntimeError("Console pressed session changed")
    composer = screen._console_composer_or_none()
    if type(composer) is not ConsoleComposerBar:
        raise RuntimeError("Console original composer is unavailable")
    _require_methods(composer, _COMPOSER_METHODS)
    ack = sending.acknowledgement_for(screen)
    _require_ack(screen, ack)
    _require_methods(ack, _ACK_METHODS)
    callback = ack.custody_callback(row.id)
    current_draft = composer.capture_draft_snapshot()
    latest_text = composer.draft_text()
    press = client.retain_authored_press(
        screen,
        visit,
        captured.authored[0],
        captured.authored[1:],
        inputs=_detach(captured.inputs, "pressed_inputs", 0, set(), thaw=True),
    )
    if (
        latest_text,
        current_draft.generation,
        current_draft.edit_serial,
    ) != captured.authored:
        client.publish_authored_inputs(
            screen,
            visit,
            latest_text,
            (current_draft.generation, current_draft.edit_serial),
        )
    receipt = None
    terminal_seen = False
    cancelled_for_display = False
    try:
        _current(screen, client, visit)
        selection = {
            "source": "backend", "attachment_id": client._attachment_id,
            "visit": visit, "rag_source_types": list(captured.rag_sources),
        }
        task = client.submit_authored_turn(press, selection, str(uuid4()))
        receipt = await asyncio.shield(task)
        await asyncio.shield(receipt.applied)
        _current(screen, client, visit)
        _require_ack(screen, ack)
        _require_methods(ack, _ACK_METHODS)
        ack.mark_admitted(row.id)
        index = 0
        accepted = False
        while True:
            event = await client.next_turn_event(receipt, index)
            if event is None:
                break
            index += 1
            if event.kind == "commit_planned":
                _current(screen, client, visit)
                # The original scheduler has completed its status/row/layout
                # handoff. Recheck the actual attached presentation before ACK.
                from .execution_state import refresh_selected_view

                try:
                    await refresh_selected_view(screen)
                except ConsoleRemoteFailure as error:
                    _require_methods(client, _PLAN_FAILURE_METHODS)
                    try:
                        cancelled = client.cancel_unacknowledged_plan(
                            receipt, event, error, screen, visit,
                        )
                        await asyncio.shield(cancelled.received)
                        try:
                            applied = await asyncio.shield(cancelled.applied)
                            if applied.get("applied") is not True:
                                raise RuntimeError("Console plan cancellation was not applied")
                        finally:
                            if await asyncio.shield(cancelled.retired) is not True:
                                raise RuntimeError("Console plan cancellation retirement is unproven")
                    except ConsoleRemoteFailure as refusal:
                        # An unqualified display or refused CancelTurn is not a
                        # known Send refusal; the outer friendly catch must not
                        # discard its original wait or claim clean settlement.
                        raise RuntimeError("Console plan cancellation was not confirmed") from refusal
                    cancelled_for_display = True
                    continue  # Observe this ReceiveTurn's own terminal/Retired; never ACK.
                _current(screen, client, visit)
                _require_ack(screen, ack)
                acknowledged = client.acknowledge_plan(receipt, event)
                await asyncio.shield(acknowledged.applied)
                if await asyncio.shield(acknowledged.retired) is not True:
                    raise ConsoleChannelLost(
                        "Console plan acknowledgement did not retire"
                    )
            elif event.kind == "input_projected":
                _current(screen, client, visit)
                _require_methods(composer, _COMPOSER_METHODS)
                if screen._console_visible_draft_session_id == row.id:
                    sent_draft.take_out_visible_sent_draft(
                        composer,
                        captured.stash,
                        lambda text: screen.app_instance.notify(
                            text, severity="warning"
                        ),
                    )
            elif event.kind == "terminal":
                terminal_seen = True
                accepted = event.payload["accepted"]
                if cancelled_for_display and accepted:
                    raise RuntimeError("Console cancelled plan unexpectedly committed")
                if callback is not None:
                    _require_ack(screen, ack)
                    _require_methods(ack, _ACK_METHODS)
                    callback(accepted)
            elif event.kind == "failed":
                # The original terminal callback remains authoritative for a
                # durable acceptance followed by a later provider failure.
                continue
        if await asyncio.shield(receipt.retired) is not True or not terminal_seen:
            raise ConsoleChannelLost("Console turn outcome or retirement is unproven")
        if client._view is screen and client._visit == visit:
            from .execution_state import refresh_selected_view

            await refresh_selected_view(screen)
        if cancelled_for_display:
            screen.app_instance.notify(
                "Console input changed; your draft was kept.", severity="warning"
            )
        return accepted
    except ConsoleReadinessExpired:
        if receipt is not None or press.task is not None:
            raise
        if client._view is not screen or client._visit != visit:
            return False
        _current(screen, client, visit)
        screen.app_instance.notify(
            "Console input changed; your draft was kept.", severity="warning"
        )
        return False
    except ConsoleRemoteFailure:
        screen.app_instance.notify(
            "Console input changed; your draft was kept.", severity="warning"
        )
        return False
    except ConsoleChannelLost:
        if client._view is not screen or client._visit != visit:
            # Navigation revokes only the old presentation. The original turn
            # and any unacknowledged plan remain owned by the actual client.
            return False
        raise
    finally:
        if client._input_press is press and press.task is None:
            client.release_authored_press(press)


_COMPOSER_METHODS = tuple(
    (name, _definition(getattr(ConsoleComposerBar, name)))
    for name in (
        "capture_draft_for_send",
        "capture_draft_snapshot",
        "draft_text",
        "commit_captured_draft",
        "spend_captured_draft",
        "retire_captured_drafts",
    )
)
_ACK_METHODS = tuple(
    (name, _definition(getattr(sending.ConsoleSendAcknowledgement, name)))
    for name in ("custody_callback", "mark_admitted", "_custody_ended", "release")
)
_PLAN_FAILURE_METHODS = tuple(
    (name, _definition(getattr(ConsoleExecutionClient, name)))
    for name in ("cancel_unacknowledged_plan", "require_passive_projection_failure", "submit")
)
_SOURCES = tuple(
    (module, name, _definition(getattr(module, name)))
    for module, names in (
        (composer_source, ("classify_console_raw_draft",)),
        (sending, ("_schedule_visible_send_capture", "acknowledgement_for")),
        (screen_source, ("_console_library_rag_source_scope",)),
        (
            sent_draft,
            ("take_out_visible_sent_draft", "commit_capture", "as_typed", "_reloaded"),
        ),
    )
    for name in names
)
_DEFINITIONS = tuple(
    (name, _definition(value))
    for name, value in tuple(globals().items())
    if type(value) is FunctionType and value.__module__ == __name__
)
