"""Publish and accept first-match activation through the existing list owner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.css.query import QueryError

from tldw_chatbook.UI.Persona_Modules.roleplay_list_window_state import ListFence
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    PersonasLibraryPane,
    option_id_for,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    DeferredPersonaEntitySelected,
    PersonaEntitySelected,
)

if TYPE_CHECKING:
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen


def land_first_match(screen: PersonasScreen, fence: ListFence) -> None:
    """Fulfil a search only at its actual current row publication."""
    window = screen._window_state
    if not (intent := window.take_first(fence)):
        window.landed_elsewhere(fence)
        return
    try:
        library = screen.query_one(PersonasLibraryPane)
    except QueryError:
        window.read_failed()
        return
    library.land_on_first()
    if not intent.activate:
        return
    option_id = library.item_list.highlighted_id()
    row = library.item_list.row(option_id) if option_id is not None else None
    if row is None:
        window.read_failed()
        return
    intent.target_id = option_id
    library.post_message(
        DeferredPersonaEntitySelected(
            entity_kind=row.kind,
            entity_id=row.item_id,
            entity_name=row.name,
            intent=intent,
        )
    )


def accept_selection(screen: PersonasScreen, message: PersonaEntitySelected) -> bool:
    """A queued search activation remains revocable until the selection handler."""
    if not isinstance(message, DeferredPersonaEntitySelected):
        return True
    window = screen._window_state
    target = option_id_for(message.entity_kind, message.entity_id)
    accepted = window.take_activation(message.intent, screen._list_fence(), target)
    try:
        item_list = screen.query_one(PersonasLibraryPane).item_list
    except QueryError:
        return False
    return (
        accepted
        and screen.is_attached
        and not screen._pruning
        and item_list.highlighted_id() == target
        and item_list.row(target) is not None
    )


def publish_first_match(
    screen: PersonasScreen,
    library: PersonasLibraryPane,
    fence: ListFence,
    request_generation: int | None = None,
) -> None:
    """Complete only the current attached publisher after its row-update await."""
    if (
        request_generation is not None
        and request_generation != screen._dictionary_lore_request_generation
        or not screen._list_fence_is_current(fence)
        or not screen.is_attached
        or screen._pruning
        or not library.is_attached
        or library._pruning
    ):
        return
    land_first_match(screen, fence)
