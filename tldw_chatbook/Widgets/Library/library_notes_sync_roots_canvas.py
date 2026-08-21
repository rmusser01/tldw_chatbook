"""Flat, message-only lasting-sync root management and recovery canvas."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncRootRow,
    LibraryNotesLastingSyncSnapshot,
)


class LibraryNotesSyncRootsCanvas(Vertical):
    """Render path-free roots with explicit, contextual next actions."""

    class RootActionRequested(Message):
        def __init__(self, root_id: str, action: str) -> None:
            super().__init__()
            self.root_id = root_id
            self.action = action

    class PageRequested(Message):
        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    class BackRequested(Message):
        pass

    def __init__(
        self, snapshot: LibraryNotesLastingSyncSnapshot, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.snapshot = snapshot
        self.add_class("library-notes-lasting-sync-canvas")

    def compose(self) -> ComposeResult:
        yield Static(
            "Library notes · Manage sync folders",
            id="notes-sync-roots-authority",
            classes="destination-section",
            markup=False,
        )
        yield Static(
            self.snapshot.status_line,
            id="notes-sync-roots-status",
            classes="library-notes-lasting-status",
            markup=False,
        )
        with VerticalScroll(id="notes-sync-roots-body"):
            if not self.snapshot.roots:
                yield Static(
                    "No lasting sync folders. Nearest valid action: Back to Notes.",
                    classes="destination-purpose",
                    markup=False,
                )
            for index, root in enumerate(self.snapshot.roots):
                with Vertical(
                    id=f"notes-sync-root-row-{index}",
                    classes="library-notes-sync-root-row",
                ):
                    yield Static(
                        root.display_name,
                        classes="destination-section",
                        markup=False,
                    )
                    yield Static(
                        f"{root.status_label} · Next: {root.next_action_label}",
                        classes="library-notes-sync-root-status",
                        markup=False,
                    )
                    with Vertical(classes="library-notes-sync-root-actions"):
                        yield from self._compose_root_actions(index, root)
            if self.snapshot.root_page_count > 1:
                yield Static(
                    f"Page {self.snapshot.root_page} of {self.snapshot.root_page_count}",
                    id="notes-sync-roots-page-status",
                    markup=False,
                )
                with Horizontal(classes="ds-toolbar"):
                    yield Button(
                        "Previous",
                        id="notes-sync-roots-page-previous",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=self.snapshot.root_page <= 1,
                    )
                    yield Button(
                        "Next",
                        id="notes-sync-roots-page-next",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=self.snapshot.root_page
                        >= self.snapshot.root_page_count,
                    )
        yield Static(
            "Retarget and Disconnect are unavailable in this release. No files or notes are changed.",
            id="notes-sync-disconnect-copy",
            classes="library-disabled-reason",
            markup=False,
        )
        with Horizontal(id="notes-sync-roots-pinned-actions", classes="ds-toolbar"):
            yield Button(
                "Back to Notes",
                id="notes-sync-roots-back",
                classes="library-canvas-action",
                compact=True,
            )

    def _compose_root_actions(
        self, index: int, root: LastingSyncRootRow
    ) -> ComposeResult:
        """Put the runtime-declared contextual action before management."""

        check_blocked = root.status in {"offline", "passive"}
        actions: list[tuple[str, str, bool, str | None]] = [
            (
                "check",
                "○ Check changes" if check_blocked else "Check changes",
                check_blocked,
                "Reconnect the folder before checking changes."
                if root.status == "offline"
                else "Use the active process to check changes."
                if root.status == "passive"
                else None,
            )
        ]
        if root.next_action == "review_changes":
            actions.append(("review", "Review", False, None))
        if root.next_action == "review_migration":
            actions.append(("migration", "Review migration", False, None))
        if root.status == "paused":
            actions.append(("resume", "Resume", False, None))
        elif root.status not in {"passive", "offline"}:
            actions.append(("pause", "Pause", False, None))
        if root.status in {"failed", "partial", "needs_attention"}:
            actions.append(("recover", "Recovery", False, None))
        actions.extend(
            (
                (
                    "retarget",
                    "○ Retarget",
                    True,
                    "Retarget is unavailable in this release.",
                ),
                (
                    "disconnect",
                    "○ Disconnect",
                    True,
                    "Disconnect is unavailable in this release.",
                ),
            )
        )
        primary_action = {
            "sync_now": "check",
            "review_changes": "review",
            "review_migration": "migration",
            "resume_sync": "resume",
            "resolve_cleanup": "recover",
            "reconnect_folder": "retarget",
            "review_settings": "retarget",
        }.get(root.next_action)
        actions.sort(key=lambda action: action[0] != primary_action)
        for action, label, disabled, tooltip in actions:
            yield self._action_button(
                index,
                root.root_id,
                action,
                label,
                disabled=disabled,
                tooltip=tooltip,
                primary=action == primary_action,
            )

    @staticmethod
    def _action_button(
        index: int,
        root_id: str,
        action: str,
        label: str,
        *,
        disabled: bool = False,
        tooltip: str | None = None,
        primary: bool = False,
    ) -> Button:
        """Use a page-local DOM token; the opaque root stays in ``name``."""

        button = Button(
            label,
            name=root_id,
            id=f"notes-sync-root-{action}-{index}",
            classes=(
                "library-canvas-action console-action-primary"
                if primary
                else "library-canvas-action"
            ),
            compact=True,
            disabled=disabled,
            tooltip=tooltip,
        )
        return button

    def sync_state(self, snapshot: LibraryNotesLastingSyncSnapshot) -> None:
        self.snapshot = snapshot
        self.refresh(recompose=True)

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "notes-sync-roots-back":
            self.post_message(self.BackRequested())
            return
        if button_id == "notes-sync-roots-page-previous":
            self.post_message(self.PageRequested(-1))
            return
        if button_id == "notes-sync-roots-page-next":
            self.post_message(self.PageRequested(1))
            return
        prefix = "notes-sync-root-"
        if not button_id.startswith(prefix):
            return
        action = button_id.removeprefix(prefix).rsplit("-", 1)[0]
        self.post_message(self.RootActionRequested(event.button.name or "", action))


__all__ = ["LibraryNotesSyncRootsCanvas"]
