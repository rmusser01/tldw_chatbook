"""App-level character change notifications (TASK-32954, spec §3.6).

A character card can be edited from the Console (via the character tools)
while the Personas workbench has that same character open. ``Character
ToolService`` posts ``CharacterCardChanged`` to the app after a committed
save so the Personas screen can refresh -- see ``ConsoleChatController.
_character_wiring`` (``Chat/console_chat_controller.py``) for the producer
and ``PersonasScreen._on_character_card_changed`` for the consumer.
"""

from __future__ import annotations

from textual.message import Message


class CharacterCardChanged(Message):
    """A character card was saved outside the Personas editor."""

    def __init__(self, character_id: int) -> None:
        super().__init__()
        self.character_id = int(character_id)
