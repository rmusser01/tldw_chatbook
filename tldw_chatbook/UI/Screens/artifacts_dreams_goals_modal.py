"""Dreams goals & privacy modal (Dreams Phase 2, Task 1).

The profile's goal rows -- persistent wants that drive event/deal/social
query angles -- with the per-goal privacy switch: a goal whose
``searchable`` flag is 0 NEVER contributes text to an outbound query or
prompt (spec §Privacy; the gate itself lives in ``query_synthesis``, this
modal only flips the stored flag). Also renders the region line,
read-only, labeled with where it is used.

Structure follows this stream's own modal idioms (see
``artifacts_dreams_modal``):

- Dismiss pattern and literal-text discipline from the story modal: goal
  text is user-typed, so every row renders through ``rich.text.Text``,
  never a bare ``str`` nor markup-parsed.
- Panel styling uses central design tokens and the existing consolidated
  CSS builder (ADR-150), with SafeModalDismissMixin for dismissal and
  opener focus restoration (ADR-161).
- All writes are single-row instant SQLite calls made directly in the
  action handlers (controller ruling, mirroring the kept-briefings and
  story modals): ``DreamsDB.upsert_profile_entry`` /
  ``delete_profile_entry`` are one-statement transactions against a
  thread-local held connection, so no worker is warranted. After ANY
  mutating action the modal calls ``on_changed()`` (the Library reader refreshes after modal return) and posts a dismissible notice; failures surface as
  notices, never crashes.

Key routing note: the goals ``ListView`` holds focus on mount so the
single-letter bindings (``a``/``x``/``s``/``q``, ADR-031) reach the screen;
the add ``Input`` consumes printable keys only while IT is focused, and
``a`` with a filled input submits without focusing it first.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, ListItem, ListView, Static

from ...Utils.input_validation import validate_dreams_profile_text
from ...Widgets.modal_dismissal import SafeModalDismissMixin

#: Weight for goal rows: goals never decay and never carry signal weight,
#: so every write uses the same nominal value.
_GOAL_WEIGHT = 1.0

_NO_DB_NOTICE = "Dreams is unavailable in this runtime: no dreams database."


class DreamsGoalsModal(SafeModalDismissMixin, ModalScreen[None]):
    """List, add, remove, and gate the profile's goals; dismisses ``None``.

    Args:
        dreams_db_getter: Zero-arg callable returning the app's
            ``DreamsDB`` or ``None``; called lazily at action time so a
            runtime without the database still renders the modal.
        on_changed: Zero-arg callback fired after every mutating action;
            the caller (the story modal / Artifacts screen) re-reads its
            Dreams rows and previews.
    """

    BINDINGS = (
        ("a", "add", "Add"),
        ("x", "remove", "Remove"),
        ("s", "toggle_searchable", "Searchable on/off"),
        ("q", "close", "Close"),
        # Hidden: Escape is the safe-dismissal grammar (task-16211), not a
        # footer-advertised action -- the hint line stays exactly the four.
        Binding("escape", "close", "Close", show=False),
    )

    SAFE_MODAL_CONTENT = "#dgm-dialog"

    # Central tokens are resolved by the existing consolidated CSS builder.
    BUNDLED_CSS = """
    DreamsGoalsModal { align: center middle; background: $background $ds-opacity-dim; }
    DreamsGoalsModal > VerticalScroll {
        width: $ds-size-76; max-width: $ds-percent-96; height: auto; max-height: $ds-percent-90;
        background: $ds-surface-raised; border: solid $ds-action-focus; padding: $ds-space-1 $ds-space-2;
    }
    DreamsGoalsModal #dgm-region { margin-top: $ds-space-1; }
    DreamsGoalsModal #dgm-goals { height: auto; max-height: $ds-size-12; margin-top: $ds-space-1; }
    DreamsGoalsModal #dgm-new-goal { margin-top: $ds-space-1; }
    DreamsGoalsModal #dgm-hints { color: $ds-status-warning; margin-top: $ds-space-1; }
    """

    def __init__(
        self,
        *,
        dreams_db_getter: Callable[[], Any],
        on_changed: Callable[[], None],
    ) -> None:
        super().__init__()
        self._dreams_db_getter = dreams_db_getter
        self._on_changed = on_changed
        self._goals: list[dict] = []

    # --- Compose ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Build the dialog: region line, goals list, add input, hint line.

        The list starts empty; :meth:`on_mount` fills it from the database
        so a broken read degrades to the empty-state hint instead of
        failing composition.

        Returns:
            The dialog's widgets, in display order.
        """
        with VerticalScroll(id="dgm-dialog"):
            yield Static("", id="dgm-region")
            yield ListView(id="dgm-goals")
            yield Static(
                Text("No goals yet -- press a, type one, press Enter.",
                     style="dim"),
                id="dgm-empty",
            )
            yield Input(placeholder="new goal", id="dgm-new-goal")
            # ADR-031 rule 4: the hint line advertises EXACTLY the four
            # implemented actions.
            yield Static(self._hints_text(), id="dgm-hints")

    @staticmethod
    def _hints_text() -> Text:
        hints = Text(style="dim")
        hints.append("a Add · x Remove · s Searchable on/off · q Close")
        return hints

    def on_mount(self) -> None:
        """Paint the region line and the goal rows, then focus the list.

        Returns:
            None.
        """
        self._load_region()
        self._reload()
        # The list (not the Input) holds focus so the single-letter
        # bindings fire; a focused Input would swallow printable keys.
        self.query_one("#dgm-goals", ListView).focus()

    def _load_region(self) -> None:
        """Paint the read-only region line (region editing is TASK-32903)."""
        try:
            # Lazy Dreams import (same pattern as the story modal).
            # The region Dreams actually USES: a stored value that fails
            # the text rule is ignored, and this line says "used in
            # queries" (review of #2921, round 3, item 7).
            from ...Dreams.settings import dreams_region

            region = dreams_region()
        except Exception:  # noqa: BLE001 - a broken config read is no region
            region = ""
        line = Text()
        line.append("Region (used in queries): ", style="dim")
        line.append(region if region else "(not set)")
        self.query_one("#dgm-region", Static).update(line)

    def _reload(self) -> None:
        """Re-read the goal rows and repaint the list, keeping selection."""
        db = self._db()
        rows: list[dict] = []
        if db is not None:
            try:
                rows = [row for row in db.list_profile()
                        if row.get("facet") == "goal"]
            except Exception as exc:  # noqa: BLE001 - a broken read keeps the modal
                logger.warning(f"Dreams goals read failed: {type(exc).__name__}")
        self._goals = rows
        list_view = self.query_one("#dgm-goals", ListView)
        previous = list_view.index
        list_view.clear()
        for row in rows:
            marker = ("searchable"
                      if int(row.get("searchable") or 0) else "private")
            line = Text()
            line.append(str(row.get("text") or ""))
            line.append(f" · {marker}", style="dim")
            list_view.append(ListItem(Static(line)))
        self.query_one("#dgm-empty", Static).display = not rows
        if rows:
            list_view.index = min(previous if previous is not None else 0,
                                  len(rows) - 1)
        else:
            list_view.index = None

    # --- Helpers -----------------------------------------------------------

    def _db(self) -> Any:
        try:
            return self._dreams_db_getter()
        except Exception:  # noqa: BLE001 - a broken getter is a missing DB
            return None

    def _selected_goal(self) -> dict | None:
        if not self._goals:
            return None
        index = self.query_one("#dgm-goals", ListView).index
        if index is None or not 0 <= index < len(self._goals):
            return None
        return self._goals[index]

    def _input(self) -> Input:
        return self.query_one("#dgm-new-goal", Input)

    def _changed(self) -> None:
        try:
            self._on_changed()
        except Exception as exc:  # noqa: BLE001 - a broken refresh must not crash
            logger.warning(f"Dreams on_changed callback failed: {type(exc).__name__}")

    # --- Actions (ADR-031 single letters) ----------------------------------

    def action_add(self) -> None:
        """Add the input's text as a user goal (searchable by default)."""
        value = self._input().value.strip()
        if not value:
            self._input().focus()
            self.notify("Type a goal first.", markup=False)
            return
        try:
            # Qodo #8 (PR #2921): goal text reaches query synthesis and
            # outbound search, so it is bounded and checked at the entry
            # point rather than persisted after a bare strip().
            value = validate_dreams_profile_text(value)
        except ValueError as exc:
            self._input().focus()
            self.notify(
                f"Goal not added: {exc}", severity="warning", markup=False
            )
            return
        db = self._db()
        if db is None:
            self.notify(_NO_DB_NOTICE, severity="warning", markup=False)
            return
        try:
            db.upsert_profile_entry(
                "goal", value, weight=_GOAL_WEIGHT, searchable=1, source="user"
            )
        except Exception as exc:  # noqa: BLE001 - a failed write is a notice
            logger.warning(f"Dreams goal add failed: {type(exc).__name__}")
            self.notify(
                f"Could not add goal: {type(exc).__name__}",
                severity="error", markup=False,
            )
            return
        self._input().value = ""
        self._reload()
        self.notify("Goal added.", markup=False)
        self._changed()

    def action_remove(self) -> None:
        """Delete the selected goal row."""
        row = self._selected_goal()
        if row is None:
            self.notify("Select a goal first.", markup=False)
            return
        db = self._db()
        if db is None:
            self.notify(_NO_DB_NOTICE, severity="warning", markup=False)
            return
        try:
            db.delete_profile_entry(int(row["id"]))
        except Exception as exc:  # noqa: BLE001 - a failed write is a notice
            logger.warning(f"Dreams goal remove failed: {type(exc).__name__}")
            self.notify(
                f"Could not remove goal: {type(exc).__name__}",
                severity="error", markup=False,
            )
            return
        self._reload()
        self.notify("Goal removed.", markup=False)
        self._changed()

    def action_toggle_searchable(self) -> None:
        """Flip the selected goal's searchable flag (the privacy gate)."""
        row = self._selected_goal()
        if row is None:
            self.notify("Select a goal first.", markup=False)
            return
        db = self._db()
        if db is None:
            self.notify(_NO_DB_NOTICE, severity="warning", markup=False)
            return
        new_flag = 0 if int(row.get("searchable") or 0) else 1
        try:
            # Upsert keyed on (facet, text): same text, flipped flag, the
            # row's own weight/source carried through unchanged.
            db.upsert_profile_entry(
                "goal", str(row["text"]),
                weight=float(row.get("weight", _GOAL_WEIGHT)),
                searchable=new_flag,
                source=str(row.get("source") or "user"),
            )
        except Exception as exc:  # noqa: BLE001 - a failed write is a notice
            logger.warning(
                f"Dreams goal searchable toggle failed: {type(exc).__name__}"
            )
            self.notify(
                f"Could not toggle searchable: {type(exc).__name__}",
                severity="error", markup=False,
            )
            return
        self._reload()
        self.notify(
            "Searchable -- this goal's text may appear in Dreams queries."
            if new_flag
            else "Private -- this goal's text never leaves this machine.",
            markup=False,
        )
        self._changed()

    def action_close(self) -> None:
        """Dismiss the modal with ``None`` (``q`` and Escape).

        Closing writes nothing: every edit was already persisted, and
        reported through ``on_changed``, by the action that made it.

        Returns:
            None.
        """
        self.dismiss(None)

    # --- Event routing ----------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the add input is the same action as ``a``."""
        event.stop()
        self.action_add()
