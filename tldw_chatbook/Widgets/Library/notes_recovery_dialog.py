"""Display one immutable local Notes pairing review; owners grant approval."""

from collections.abc import Awaitable, Callable
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Notes.recovery_review import NotesRecoveryReview
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


class NotesRecoveryDialog(ModalScreen[None]):
    """Inspect an owner dry-run and request a separately confirmed approval."""

    BINDINGS: ClassVar = [("escape", "close", "Close")]
    DEFAULT_CSS = """
    NotesRecoveryDialog { align: center middle; }
    #notes-recovery-dialog { width: 85%; max-width: 100; height: 85%;
        border: round $accent; background: $surface; padding: 1 2; }
    #notes-recovery-rows { height: 1fr; }
    #notes-recovery-actions { height: auto; }
    #notes-recovery-dialog Static { height: auto; margin-bottom: 1; }
    """

    def __init__(
        self,
        review: NotesRecoveryReview,
        *,
        current: Callable[[], bool],
        approve: Callable[[NotesRecoveryReview], Awaitable[object]],
    ) -> None:
        super().__init__()
        self.review = review
        self._current = current
        self._approve = approve
        self._busy = False
        self._finished = False

    def compose(self) -> ComposeResult:
        label = (
            "Notes Sync" if self.review.owner == "notes.sync_bindings" else "File Notes"
        )
        with Vertical(id="notes-recovery-dialog"):
            yield Static(f"Review recovered {label} pairing", markup=False)
            with VerticalScroll(id="notes-recovery-rows"):
                yield Static(
                    str(self.review.root), id="notes-recovery-root", markup=False
                )
                yield Static(
                    "\n".join(
                        f"{path} — {state}" for path, state in self.review.entries
                    )
                    or "No local or retained files in this comparison.",
                    id="notes-recovery-entries",
                    markup=False,
                )
                yield Static(
                    "Issues:\n" + ("\n".join(self.review.issues) or "None"),
                    id="notes-recovery-issues",
                    markup=False,
                )
                yield Static(
                    "Historical managed memberships remain inactive; approval does not "
                    "transfer them or replay old filesystem intents.\n"
                    + (
                        "\n".join(self.review.historical_owners)
                        or "No historical owners."
                    ),
                    id="notes-recovery-history",
                    markup=False,
                )
            yield Static(
                "Approval rechecks this comparison. Sync and Refresh remain separate actions.",
                id="notes-recovery-status",
                markup=False,
            )
            with Horizontal(id="notes-recovery-actions"):
                yield Button("Close", id="notes-recovery-close")
                yield Button(
                    "Approve pairing…",
                    id="notes-recovery-approve",
                    disabled=bool(self.review.issues),
                )

    def on_mount(self) -> None:
        self.set_interval(0.25, self._check_current)

    def _check_current(self) -> bool:
        if self._finished:
            return False
        if not self._current():
            self.query_one("#notes-recovery-approve", Button).disabled = True
            self.query_one("#notes-recovery-status", Static).update(
                "Selection changed. Close this review and review the current folder again."
            )
            return False
        return True

    @on(Button.Pressed, "#notes-recovery-approve")
    def request_approval(self, event: Button.Pressed) -> None:
        event.stop()
        if self._busy or self.review.issues or not self._check_current():
            return
        self._busy = True
        self.run_worker(self._confirm(), group="notes-pairing-confirm")

    async def _confirm(self) -> None:
        from sqlite3 import Error as SQLiteError

        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDBError

        try:
            confirmed = await self.app.push_screen_wait(
                ConfirmationDialog(
                    title="Approve this local Notes pairing?",
                    message="Recheck the reviewed folder and retained notes, then approve only "
                    "this pairing. This does not run Sync or Refresh.",
                    confirm_label="Approve pairing",
                )
            )
            if not confirmed or not self._check_current():
                return
            self.query_one("#notes-recovery-approve", Button).disabled = True
            await self._approve(self.review)
            self._finished = True
            self.query_one("#notes-recovery-status", Static).update(
                "Pairing approved. Sync or Refresh can be requested separately."
            )
        except (
            OSError,
            ValueError,
            RuntimeError,
            SQLiteError,
            CharactersRAGDBError,
        ) as error:
            self._finished = True
            self.query_one("#notes-recovery-approve", Button).disabled = True
            self.query_one("#notes-recovery-status", Static).update(
                f"Pairing was not approved: {error}. Close and review again."
            )
        finally:
            self._busy = False

    @on(Button.Pressed, "#notes-recovery-close")
    def close_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_close()

    def action_close(self) -> None:
        self.dismiss(None)
