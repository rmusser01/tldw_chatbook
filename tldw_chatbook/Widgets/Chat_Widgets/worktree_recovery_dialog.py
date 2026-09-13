"""Metadata-only recorded work picker; selecting an action never consents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Agents.agent_models import TERMINAL_RUN_STATUSES

from .worktree_confirm_card import plain


def row_status(row: Mapping[str, Any]) -> tuple[bool, str]:
    """Distinguish positive completion from absent/uncertain writer proof."""
    state = row.get("mutation_state")
    resolved = {
        "applied": "Applied; source checkout retained.",
        "merged": "Merged; source checkout retained.",
        "discarded_cleanup_pending": "Discarded; baseline checkout retained.",
    }
    if state in resolved:
        return False, resolved[state]
    if row.get("writer_state") == "held":
        return False, "Completion has not been confirmed; work is retained."
    if row.get("writer_state") != "drained" or state != "unresolved":
        return False, "The result needs manual review; work is retained."
    if row.get("run_status") not in TERMINAL_RUN_STATUSES:
        return False, "The run has not finished; work is retained."
    return True, "Ready for review. Each action asks for confirmation."


class _RecoveryButton(Button):
    def __init__(self, run_id, action, *, disabled):
        super().__init__(action.title(), disabled=disabled)
        self.run_id = run_id
        self.action = action


class WorktreeRecoveryDialog(ModalScreen):
    """Scrollable recorded work with fixed close/pagination controls."""

    BINDINGS = [("escape", "close", "Close")]  # noqa: RUF012 - Textual binding declaration

    def __init__(self, page, *, busy=False):
        super().__init__()
        self.page = page
        self.busy = busy

    def compose(self):
        with Vertical(id="worktree-recovery-dialog"):
            yield Static(
                "Recover agent work", classes="worktree-recovery-title", markup=False
            )
            yield Static(
                plain(self.page.conversation_id)
                + "\nRepository: "
                + plain(self.page.repository),
                classes="worktree-recovery-context",
                markup=False,
            )
            with VerticalScroll(id="worktree-recovery-rows"):
                if self.page.message:
                    yield Static(plain(self.page.message), markup=False)
                elif not self.page.rows:
                    yield Static(
                        "No recorded agent work for this conversation and repository.",
                        markup=False,
                    )
                for row in self.page.rows:
                    enabled, reason = row_status(row)
                    with Vertical(classes="worktree-recovery-row"):
                        yield Static(
                            "Run: "
                            + plain(row["run_id"])
                            + "\nSource: "
                            + plain(row["child_path"]),
                            markup=False,
                        )
                        yield Static(
                            "A recovery operation is in progress."
                            if self.busy
                            else reason,
                            markup=False,
                        )
                        with Horizontal(classes="worktree-recovery-actions"):
                            for action in ("apply", "merge", "discard"):
                                yield _RecoveryButton(
                                    row["run_id"],
                                    action,
                                    disabled=self.busy or not enabled,
                                )
            with Horizontal(id="worktree-recovery-footer"):
                yield Button(
                    "Next page",
                    id="worktree-next",
                    disabled=self.page.next_run_id is None,
                )
                yield Button("Close", id="worktree-close")

    def action_close(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed)
    def selected(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.disabled:
            return
        if isinstance(event.button, _RecoveryButton):
            self.dismiss((event.button.run_id, event.button.action))
        elif event.button.id == "worktree-next":
            self.dismiss((self.page.next_run_id, "next"))
        else:
            self.dismiss(None)
