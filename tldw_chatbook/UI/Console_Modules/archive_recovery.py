"""Disposable Console adapter for runtime-owned archive recovery state."""

from __future__ import annotations

import asyncio
from typing import Any

from textual import on
from textual.containers import Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Chat.archive_recovery.contracts import (
    ArchiveRecoveryError,
    RecoveryConfirmation,
    RecoveryOperationSnapshot,
)
from tldw_chatbook.Chat.archive_recovery.runtime import (
    ArchiveRecoveryPresentation,
    ArchiveRecoveryRuntime,
)
from tldw_chatbook.Widgets.Chat_Widgets.archive_recovery_cards import (
    ArchiveRecoveryConfirmCard,
    ArchiveRecoveryResultsCard,
    RecoveryActionButton,
    safe_text,
)


class ConsoleArchiveRecoverySurface(Vertical):
    """Mounted message routing stays within the disposable recovery surface."""

    def __init__(self, view: ConsoleArchiveRecoveryView) -> None:
        super().__init__(id="console-archive-recovery-controls")
        self.view = view

    async def on_archive_recovery_confirm_card_recovery_decided(self, event):
        event.stop()
        await self.view.decide(event)

    def on_archive_recovery_confirm_card_rename_requested(self, event):
        event.stop()
        self.view.rename(event.session_id, event.request_id)

    async def on_archive_recovery_results_card_action_requested(self, event):
        event.stop()
        await self.view.action(event)

    @on(Button.Pressed, "#recovery-manual-library")
    def imported_library_search(self, event):
        event.stop()
        source = getattr(event.button, "source_session_id", None)

        self.app.run_worker(
            self.view.open_library(session_id=source), group="archive-recovery-library"
        )

    @on(
        Button.Pressed,
        ".recovery-receipt-open, .recovery-receipt-rename, .recovery-receipt-refresh",
    )
    async def receipt_pressed(self, event):
        await self.view.receipt_action(event)


class ConsoleArchiveRecoveryView:
    """Project trusted records; coordinator remains the sole decision/effect owner."""

    def __init__(self, screen: Any) -> None:
        self.screen = screen
        self._sync_lock = asyncio.Lock()
        self._session_id = None
        self._render_generation = 0
        self._rendered_sets: set[str] = set()
        self._rendered_receipts: set[str] = set()
        self._receipt_actions: dict[str, RecoveryOperationSnapshot] = {}

    def runtime(self) -> ArchiveRecoveryRuntime:
        return self.screen._console_runtime().ensure_archive_recovery()

    def _surface(self) -> ConsoleArchiveRecoverySurface | None:
        if not hasattr(self.screen, "_nodes"):
            return None
        matches = self.screen.query("#console-archive-recovery-controls")
        return matches.first() if matches else None

    def _confirmation(self) -> ArchiveRecoveryConfirmCard | None:
        surface = self._surface()
        if surface is None:
            return None
        cards = surface.query(ArchiveRecoveryConfirmCard)
        return cards.first() if cards else None

    def is_answerable(self, session_id: str) -> bool:
        """Require the current mounted source and already mounted exact controls."""
        store = self.screen._console_chat_store
        card = self._confirmation()
        return bool(
            self.screen._console_runtime().view is self.screen
            and self.screen.is_mounted
            and self.screen.is_attached
            and self.screen.is_current
            and store
            and store.active_session_id == session_id
            and card
            and card.is_mounted
            and card.is_attached
            and all(
                bool(card.query(f"#{identity}"))
                and card.query(f"#{identity}").first().is_mounted
                and card.query(f"#{identity}").first().is_attached
                for identity in (
                    "recovery-confirm",
                    "recovery-cancel",
                    "recovery-scope",
                )
            )
        )

    async def sync(self) -> None:
        """Remount only runtime custody, never persisted transcript dictionaries."""
        async with self._sync_lock:
            if not hasattr(self.screen, "_nodes"):
                return
            roots = self.screen.query("#console-archive-recovery")
            if not roots:
                return
            root = roots.first()
            if not root.is_mounted or not root.is_attached:
                return
            surface = self._surface()
            if surface is None:
                surface = ConsoleArchiveRecoverySurface(self)
                await root.mount(surface)
            if not root.is_attached or not surface.is_attached:
                return
            root.display = True
            store = self.screen._console_chat_store
            if not store:
                return
            runtime = self.runtime()
            session_id = store.active_session_id
            if not session_id:
                return
            if self._session_id != session_id or self._confirmation() is None:
                self._render_generation += 1
                await surface.remove_children()
                if not surface.is_attached:
                    return
                self._rendered_sets.clear()
                self._rendered_receipts.clear()
                self._receipt_actions.clear()
                self._session_id = session_id
                await surface.mount(ArchiveRecoveryConfirmCard())
            presentations = runtime.presentations(session_id)
            live_sets = {record.projection["result_set_id"] for record in presentations}
            for card in list(surface.query(ArchiveRecoveryResultsCard)):
                if card.record.projection["result_set_id"] not in live_sets:
                    await card.remove()
            self._rendered_sets.intersection_update(live_sets)
            for record in presentations:
                await self._mount_results(record)
            if not surface.is_attached:
                return
            imported = not presentations and any(
                "result_set_id" in message.content and "result_ref" in message.content
                for message in store.messages_for_session(session_id)
            )
            if imported and not surface.query("#recovery-manual-library"):
                library_button = RecoveryActionButton(
                    "Search archived chats in Library", id="recovery-manual-library"
                )
                library_button.source_session_id = session_id
                await surface.mount(
                    Vertical(
                        Static(
                            "Saved recovery text has no live action provenance. Search again in Library.",
                            markup=False,
                            classes="archive-recovery-copy",
                        ),
                        library_button,
                        classes="archive-recovery-card",
                    )
                )
            snapshot = runtime.coordinator.snapshot(session_id)
            self.show_confirmation(snapshot.confirmation if snapshot else None)
            receipts = runtime.coordinator.receipts(session_id)
            live_receipts = {receipt.operation_id for receipt in receipts}
            for card in list(surface.query(".archive-recovery-receipt")):
                if card.name not in live_receipts:
                    await card.remove()
            self._rendered_receipts.intersection_update(live_receipts)
            self._receipt_actions = {
                key: receipt
                for key, receipt in self._receipt_actions.items()
                if receipt.operation_id in live_receipts
            }
            for receipt in receipts:
                await self._mount_receipt(receipt)
            surface.display = bool(
                self._rendered_sets
                or self._rendered_receipts
                or (snapshot and snapshot.confirmation)
                or imported
            )

    def show_results(self, projection: ArchiveRecoveryPresentation) -> None:
        """Schedule a projection; runtime retains it if the view goes away."""
        if projection.session_id != self._session_id or not self.is_answerable(
            projection.session_id
        ):
            return
        self.screen.run_worker(
            self._mount_results(projection), group="archive-recovery-projection"
        )

    async def _mount_results(self, record: ArchiveRecoveryPresentation) -> None:
        surface = self._surface()
        set_id = record.projection["result_set_id"]
        if (
            surface is None
            or record.session_id != self._session_id
            or not self.is_answerable(record.session_id)
            or set_id in self._rendered_sets
        ):
            return
        generation = self._render_generation
        card = ArchiveRecoveryResultsCard(record)
        self._rendered_sets.add(set_id)
        try:
            await surface.mount(card, before=self._confirmation())
            if (
                generation != self._render_generation
                or record.session_id != self._session_id
                or not self.is_answerable(record.session_id)
            ):
                await card.remove()
                if generation == self._render_generation:
                    self._rendered_sets.discard(set_id)
                return
            surface.display = True
        except Exception:
            self._rendered_sets.discard(set_id)
            raise

    def show_confirmation(self, confirmation: RecoveryConfirmation | None) -> bool:
        """Return true only after exact typed controls are synchronously answerable."""
        card = self._confirmation()
        if card is None or not card.is_mounted:
            return False
        if confirmation is not None and not self.is_answerable(confirmation.session_id):
            return False
        card.set_confirmation(confirmation)
        surface = self._surface()
        if confirmation is not None and surface is not None:
            surface.display = True
        return (
            confirmation is not None
            and card.confirmation is confirmation
            and card.query_one("#recovery-cancel").is_mounted
        )

    async def show_receipt(self, outcome: RecoveryOperationSnapshot) -> bool:
        """Acknowledge actual source-bound projection to the application owner."""
        if outcome.session_id != self._session_id or not self.is_answerable(
            outcome.session_id
        ):
            return False
        card = self._confirmation()
        focused = self.screen.focused
        expired_focus = bool(
            outcome.outcome
            and outcome.outcome.status == "expired"
            and card
            and focused
            and (focused is card or card in focused.ancestors)
        )
        self.show_confirmation(None)
        if expired_focus:
            buttons = self.screen.query("#recovery-refresh")
            if buttons:
                buttons.first().focus()
        return await self._mount_receipt(outcome)

    async def _mount_receipt(self, receipt: RecoveryOperationSnapshot) -> bool:
        surface = self._surface()
        if (
            surface is None
            or receipt.session_id != self._session_id
            or not receipt.outcome
            or not self.is_answerable(receipt.session_id)
        ):
            return False
        if receipt.operation_id in self._rendered_receipts:
            return True
        generation = self._render_generation
        self._rendered_receipts.add(receipt.operation_id)
        outcome = receipt.outcome
        children = [
            Static(
                safe_text(outcome.message),
                markup=False,
                classes="archive-recovery-copy",
            )
        ]
        if outcome.chat_available or outcome.next_action == "restore_workspace":
            key = f"recovery-open-{receipt.operation_id}"
            self._receipt_actions[key] = receipt
            children.append(
                RecoveryActionButton(
                    "Open" if outcome.chat_available else "Restore workspace to open",
                    id=key,
                    classes="recovery-receipt-open",
                )
            )
        if outcome.status == "name_conflict" and receipt.confirmation:
            key = f"recovery-rename-{receipt.operation_id}"
            self._receipt_actions[key] = receipt
            children.append(
                RecoveryActionButton(
                    "Restore as…", id=key, classes="recovery-receipt-rename"
                )
            )
        if outcome.next_action == "refresh":
            key = f"recovery-refresh-{receipt.operation_id}"
            self._receipt_actions[key] = receipt
            children.append(
                RecoveryActionButton(
                    "Refresh results", id=key, classes="recovery-receipt-refresh"
                )
            )
        card = Vertical(
            *children,
            name=receipt.operation_id,
            classes="archive-recovery-card archive-recovery-receipt",
        )
        try:
            await surface.mount(card)
        except Exception:
            self._rendered_receipts.discard(receipt.operation_id)
            raise
        if (
            generation != self._render_generation
            or receipt.session_id != self._session_id
            or not self.is_answerable(receipt.session_id)
        ):
            await card.remove()
            self._rendered_receipts.discard(receipt.operation_id)
            return False
        surface.display = True
        return True

    async def decide(self, event: ArchiveRecoveryConfirmCard.RecoveryDecided) -> None:
        if not self.is_answerable(event.session_id):
            return
        await self.runtime().coordinator.decide(
            event.session_id, event.request_id, allow=event.allow, scope=event.scope
        )
        await self.sync()

    async def action(self, event: ArchiveRecoveryResultsCard.ActionRequested) -> None:
        if not self.is_answerable(event.session_id):
            return
        if event.action == "library":
            self.screen.app.run_worker(
                self.open_library(session_id=event.session_id),
                group="archive-recovery-library",
            )
            return
        runtime = self.runtime()
        record = next(
            (
                r
                for r in runtime.presentations(event.session_id)
                if r.projection["result_set_id"] == event.result_set_id
            ),
            None,
        )
        if record is None:
            self.screen.notify(
                "Recovery results expired. Search again in Library.", severity="warning"
            )
            return
        try:
            if event.action in {"more", "refresh", "unknown"}:
                arguments = (
                    {"cursor": record.projection["cursor"]}
                    if event.action == "more"
                    else record.request.model_dump(mode="json")
                )
                if event.action == "unknown":
                    arguments["include_unknown_archive_times"] = True
                await runtime.search(event.session_id, arguments, model_visible=False)
            else:
                item = runtime.results.resolve(event.session_id, event.result_ref)
                if event.action == "restore":
                    await runtime.coordinator.request(
                        event.session_id, event.result_ref, call_key=None, scope=None
                    )
                    await self.sync()
                elif event.action == "preview":
                    self.screen.app.run_worker(
                        self.open_library(
                            item.target.conversation_id, session_id=event.session_id
                        ),
                        group="archive-recovery-library",
                    )
        except ArchiveRecoveryError as exc:
            self.screen.notify(str(exc), severity="warning")

    async def open_library(
        self, conversation_id: str | None = None, *, session_id: str
    ) -> None:
        """Use the canonical reader only while the requesting source is current."""
        if not self.is_answerable(session_id):
            return
        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
        from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

        app = self.screen.app_instance
        from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_MODE

        await app.handle_screen_navigation(
            NavigateToScreen("library", {LIBRARY_NAV_CONTEXT_MODE: "conversations"})
        )
        library = self.screen.app.screen
        if isinstance(library, LibraryScreen):
            library._conversation_recovery().scope = "all"
            if conversation_id:
                await library._open_library_item_by_id("conversations", conversation_id)
            else:
                library._conversation_recovery().set_scope("all")

    def rename(self, session_id: str, request_id: str) -> None:
        if not self.is_answerable(session_id):
            return
        coordinator = self.runtime().coordinator
        snapshot = coordinator.snapshot(session_id)
        card = snapshot.confirmation if snapshot else None
        if card is None or card.request_id != request_id:
            card = next(
                (
                    r.confirmation
                    for r in coordinator.receipts(session_id)
                    if r.confirmation and r.confirmation.request_id == request_id
                ),
                None,
            )
        if card is None or card.target.workspace is None:
            return
        from tldw_chatbook.Widgets.Console.console_workspace_switcher_modal import (
            ConsoleWorkspaceRenameModal,
        )

        generation = self._render_generation

        def selected(name: str | None) -> None:
            if name:
                self.screen.run_worker(
                    self._revise_workspace_name(
                        session_id, request_id, name, generation
                    ),
                    group="archive-recovery-rename",
                )

        self.screen.app.push_screen(
            ConsoleWorkspaceRenameModal(
                current_name=card.replacement_name or card.target.workspace.name,
                restoring=True,
            ),
            selected,
        )

    async def _revise_workspace_name(
        self, session_id: str, request_id: str, name: str, generation: int
    ) -> None:
        """Keep expected modal-time refusal recoverable for its owning source."""
        if generation != self._render_generation or not self.is_answerable(session_id):
            return
        try:
            await self.runtime().coordinator.revise_workspace_name(
                session_id, request_id, name
            )
        except ArchiveRecoveryError as exc:
            if generation == self._render_generation and self.is_answerable(session_id):
                self.screen.notify(str(exc), severity="warning")
        if generation == self._render_generation and self.is_answerable(session_id):
            await self.sync()

    async def receipt_action(self, event: Button.Pressed) -> None:
        receipt = self._receipt_actions.get(event.button.id)
        if receipt is None or not self.is_answerable(receipt.session_id):
            return
        event.stop()
        if event.button.has_class("recovery-receipt-rename"):
            self.rename(receipt.session_id, receipt.confirmation.request_id)
        elif event.button.has_class("recovery-receipt-refresh"):
            records = self.runtime().presentations(receipt.session_id)
            if records:
                generation = self._render_generation
                try:
                    await self.runtime().search(
                        receipt.session_id,
                        records[-1].request.model_dump(mode="json"),
                        model_visible=False,
                    )
                except ArchiveRecoveryError as exc:
                    if generation == self._render_generation and self.is_answerable(
                        receipt.session_id
                    ):
                        self.screen.notify(str(exc), severity="warning")
        else:
            from tldw_chatbook.UI.Console_Modules.archive import (
                request_conversation_resume,
            )

            await request_conversation_resume(
                self.screen.app_instance,
                receipt.outcome.conversation_id,
                request_if=lambda: self.is_answerable(receipt.session_id),
            )
