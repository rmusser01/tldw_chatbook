"""Preserve selected rail-toggle intent while its display facts refresh."""

from collections import deque
from dataclasses import dataclass
import time

from ...Chat.console_domain_services import _definition
from ..Screens.chat_screen import ChatScreen
from .execution_state import (
    ConsoleDisplayExpired, _require_methods, refresh_selected_when_due, selected_client,
    selected_rail_publication,
)


_APPLY = tuple((name, _definition(getattr(ChatScreen, name))) for name in (
    "_apply_console_rail_section_toggle", "_prepare_console_rail_section_toggle",
    "_write_console_rail_section_toggle", "_finish_console_rail_section_toggle",
))


@dataclass
class _RailToggle:
    identity: tuple
    section_id: str
    next_open: bool | None
    submitted: bool = False


def _identity(client):
    row = client.projection_view.session(client.projection_view.active_session_id)
    return client, client._visit, row.id, row.incarnation_id


def _same_intent_scope(screen, client, visit, captured):
    client.require_view(screen, visit)
    if selected_client(screen) is not client:
        raise RuntimeError("Selected rail toggle owner changed")
    identity = _identity(client)
    return captured[0] is client and captured[1:] == identity[1:] and visit == client._visit


def apply_selected_rail_toggle(screen, section_id: str, *, next_open: bool | None) -> None:
    """Retain each actual input in order; a refresh never invents a toggle."""
    publication = selected_rail_publication(screen)
    client = publication.client if publication is not None else selected_client(screen)
    if client is None:
        raise RuntimeError("Selected rail toggle requires its actual client")
    if getattr(screen, "_console_process_content_failed", False):
        return
    if publication is not None:
        client.require_view(screen, publication.visit)
    pending = getattr(screen, "_console_pending_process_rail_toggles", None)
    if pending is None:
        pending = screen._console_pending_process_rail_toggles = deque()
    pending.append(_RailToggle(_identity(client), section_id, next_open))
    replay_selected_rail_toggles(screen, client, client._visit)


def replay_selected_rail_toggles(screen, client, visit: int) -> None:
    """Apply current queued inputs once, after the original positive refresh."""
    pending = getattr(screen, "_console_pending_process_rail_toggles", None)
    if not pending:
        return
    if getattr(screen, "_console_process_content_failed", False):
        return
    if selected_rail_publication(screen, client) is not None:
        return  # Keep both unsent and post-write intent until publication releases.
    while pending:
        intent = pending[0]
        captured = intent.identity
        if not _same_intent_scope(screen, client, visit, captured):
            pending.popleft()
            continue
        if time.monotonic() >= client._readiness_deadline:
            refresh_selected_when_due(screen)
            return
        _require_methods(screen, _APPLY)
        if not intent.submitted:
            try:
                next_open = screen._prepare_console_rail_section_toggle(
                    intent.section_id, next_open=intent.next_open,
                )
            except ConsoleDisplayExpired:
                refresh_selected_when_due(screen)
                return
            except BaseException:
                pending.popleft()
                raise
            if not _same_intent_scope(screen, client, visit, captured):
                pending.popleft()
                continue
            # Derivation may consume the remaining display lifetime. Keep the
            # unresolved gesture until a fresh pre-effect derivation succeeds.
            if time.monotonic() >= client._readiness_deadline:
                refresh_selected_when_due(screen)
                return
            try:
                screen._write_console_rail_section_toggle(
                    intent.section_id, next_open=next_open,
                )
            except BaseException:
                # An uncertain original write is never permission to retry it.
                pending.popleft()
                raise
            intent.next_open = next_open
            intent.submitted = True
        try:
            rail_state = screen._current_console_rail_state()
        except ConsoleDisplayExpired:
            refresh_selected_when_due(screen)
            return
        except BaseException:
            pending.popleft()
            raise
        if not _same_intent_scope(screen, client, visit, captured):
            pending.popleft()
            continue
        if time.monotonic() >= client._readiness_deadline:
            refresh_selected_when_due(screen)
            return
        # The original preference was already submitted. A redraw retry must
        # never recompute the gesture or repeat its preference/Agent effects.
        pending.popleft()
        screen._finish_console_rail_section_toggle(
            intent.section_id, next_open=intent.next_open, rail_state=rail_state,
        )
