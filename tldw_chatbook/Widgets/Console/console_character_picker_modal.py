"""Console character picker: choose a character, then choose where it lands.

task-1672: the Console shell strip's ``Character:``/``Assistant:`` chip is an
action chip (``ConsoleAssistantChip``). Activating it opens this modal, which
searches saved character cards and returns BOTH the chosen character and the
placement the user picked -- swap the current session, or start a fresh chat.

The two-step shape is deliberate (user decision, 2026-07-31): swapping a
character mid-conversation changes the system prompt the model sees while the
existing transcript stays on screen, and starting fresh loses the current
thread. Neither is right for every case, so the user says which, per pick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

CharacterPlacement = Literal["swap", "new"]

#: Bounds the result list so a large card library cannot mount hundreds of
#: rows into a modal (the switcher modal caps its own list the same way).
CHARACTER_PICKER_MAX_RESULTS = 40


@dataclass(frozen=True)
class ConsoleCharacterChoice:
    """A picked character plus where the user wants to use it."""

    character_id: int
    name: str
    placement: CharacterPlacement


@dataclass(frozen=True)
class ConsoleCharacterOption:
    """One selectable character row."""

    character_id: int
    name: str
    description: str = ""


def filter_character_options(
    options: Sequence[ConsoleCharacterOption],
    query: str,
    *,
    limit: int = CHARACTER_PICKER_MAX_RESULTS,
) -> tuple[ConsoleCharacterOption, ...]:
    """Filter character options by a case-insensitive substring query.

    Name matches rank above description matches so typing a character's
    name never buries it under cards that merely mention it.

    Args:
        options: All selectable characters.
        query: The raw search text; blank returns the head of the list.
        limit: Maximum rows to return.

    Returns:
        Matching options, name-matches first, bounded by ``limit``.
    """
    text = (query or "").strip().lower()
    if not text:
        return tuple(options[:limit])
    primary = [o for o in options if text in o.name.lower()]
    secondary = [
        o
        for o in options
        if text not in o.name.lower() and text in (o.description or "").lower()
    ]
    return tuple((primary + secondary)[:limit])


class ConsoleCharacterPickerModal(ModalScreen["ConsoleCharacterChoice | None"]):
    """Search saved characters, then place the pick in this chat or a new one."""

    DEFAULT_CSS = """
    ConsoleCharacterPickerModal {
        align: center middle;
    }

    #console-character-picker {
        width: 72;
        height: auto;
        max-height: 30;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-character-picker-results {
        height: auto;
        max-height: 16;
        margin: 1 0 0 0;
    }

    .console-character-picker-result {
        width: 100%;
        height: 1;
        min-height: 1;
        margin: 0;
    }

    #console-character-picker-placement {
        height: auto;
        margin: 1 0 0 0;
    }

    .console-character-picker-hint {
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("escape", "dismiss_picker", "Cancel"),
        ("down", "cursor_down", "Next"),
        ("up", "cursor_up", "Previous"),
    ]

    def __init__(
        self,
        *,
        options: Sequence[ConsoleCharacterOption],
        current_character_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the picker.

        Args:
            options: Selectable characters, already loaded off-thread by the
                caller (this modal never touches the database itself).
            current_character_id: The session's current character, marked in
                the list so the user can see what they are replacing.
            **kwargs: Forwarded to ``ModalScreen``.
        """
        super().__init__(**kwargs)
        self._options = tuple(options)
        self._current_character_id = current_character_id
        self._results: tuple[ConsoleCharacterOption, ...] = ()
        self._selected_index = 0
        self._pending: ConsoleCharacterOption | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="console-character-picker"):
            yield Static("Choose character", classes="console-modal-header")
            yield Input(
                placeholder="Search characters…",
                id="console-character-picker-query",
            )
            yield Vertical(id="console-character-picker-results")
            placement = Horizontal(id="console-character-picker-placement")
            placement.display = False
            with placement:
                yield Button("Swap in this chat", id="console-character-placement-swap")
                yield Button("Start new chat", id="console-character-placement-new")
            yield Static(
                "Enter picks a character, then choose where it lands.",
                id="console-character-picker-hint",
                classes="console-character-picker-hint",
            )

    async def on_mount(self) -> None:
        self.query_one("#console-character-picker-query", Input).focus()
        await self._refresh_results("")

    async def _refresh_results(self, query: str) -> None:
        """Recompute and remount the result rows for ``query``."""
        self._results = filter_character_options(self._options, query)
        self._selected_index = 0
        results = self.query_one("#console-character-picker-results", Vertical)
        await results.remove_children()
        if not self._results:
            await results.mount(
                Static(
                    "No characters match.",
                    id="console-character-picker-empty",
                    markup=False,
                )
            )
            return
        rows = []
        for index, option in enumerate(self._results):
            marker = "> " if index == self._selected_index else "  "
            current = "  (current)" if option.character_id == (
                self._current_character_id
            ) else ""
            rows.append(
                Static(
                    f"{marker}{option.name}{current}",
                    id=f"console-character-picker-row-{option.character_id}",
                    classes="console-character-picker-result",
                    markup=False,
                )
            )
        await results.mount(*rows)

    def _select(self, option: ConsoleCharacterOption) -> None:
        """Stage ``option`` and reveal the placement buttons."""
        self._pending = option
        placement = self.query_one("#console-character-picker-placement", Horizontal)
        placement.display = True
        self.query_one("#console-character-picker-hint", Static).update(
            f"{option.name}: swap into this chat, or start a new one?"
        )
        self.query_one("#console-character-placement-swap", Button).focus()

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "console-character-picker-query":
            return
        event.stop()
        await self._refresh_results(event.value)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "console-character-picker-query":
            return
        event.stop()
        if self._results:
            self._select(self._results[self._selected_index])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._pending is None:
            return
        if event.button.id == "console-character-placement-swap":
            event.stop()
            self._finish("swap")
        elif event.button.id == "console-character-placement-new":
            event.stop()
            self._finish("new")

    def _finish(self, placement: CharacterPlacement) -> None:
        assert self._pending is not None
        self.dismiss(
            ConsoleCharacterChoice(
                character_id=self._pending.character_id,
                name=self._pending.name,
                placement=placement,
            )
        )

    async def action_cursor_down(self) -> None:
        if self._results:
            self._selected_index = (self._selected_index + 1) % len(self._results)
            await self._repaint_markers()

    async def action_cursor_up(self) -> None:
        if self._results:
            self._selected_index = (self._selected_index - 1) % len(self._results)
            await self._repaint_markers()

    async def _repaint_markers(self) -> None:
        for index, option in enumerate(self._results):
            marker = "> " if index == self._selected_index else "  "
            current = "  (current)" if option.character_id == (
                self._current_character_id
            ) else ""
            try:
                row = self.query_one(
                    f"#console-character-picker-row-{option.character_id}", Static
                )
            except Exception:
                continue
            row.update(f"{marker}{option.name}{current}")

    def action_dismiss_picker(self) -> None:
        self.dismiss(None)
