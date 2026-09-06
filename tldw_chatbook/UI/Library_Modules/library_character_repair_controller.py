"""Library-owned presentation controller for unresolved character repair."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Select, Static

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterRepairCandidate,
    CharacterRepairPage,
    CharacterRepairRequest,
    CharacterRepairResult,
    ResolvedLocalCharacterKey,
)
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    LibraryCharacterRepairContext,
    RoleplayReturnTarget,
)


class _RepairService(Protocol):
    def repair_candidates(self, key, *, offset=0, limit=20) -> CharacterRepairPage: ...

    def repair(self, request: CharacterRepairRequest) -> CharacterRepairResult: ...

    def refresh_unresolved_evidence(self, key) -> tuple[int, str] | None: ...


@dataclass(frozen=True)
class _RepairChoices:
    context: LibraryCharacterRepairContext | None
    page: CharacterRepairPage
    revision: int | None
    restarted: bool = False


_RESULT_COPY = {
    CharacterRepairResult.STALE_VERSION: (
        "Conversation changed. Refresh before repairing."
    ),
    CharacterRepairResult.NOT_FOUND: "Conversation no longer exists.",
    CharacterRepairResult.INVALID_CANDIDATE: (
        "Selected character is no longer available. Refresh choices."
    ),
}


class LibraryCharacterRepairController:
    """Present explicit same-authority CAS repair without guessing by name."""

    def __init__(
        self,
        *,
        service: _RepairService,
        invalidate_keyword: Callable[[], None],
        invalidate_semantic: Callable[[], None],
        return_to_anchor: Callable[[RoleplayReturnTarget], None],
        focus_refresh: Callable[[], None],
        source_revision: Callable[[], int | None] = lambda: None,
    ) -> None:
        self._service = service
        self._invalidate_keyword = invalidate_keyword
        self._invalidate_semantic = invalidate_semantic
        self._return_to_anchor = return_to_anchor
        self._focus_refresh = focus_refresh
        self._source_revision = source_revision
        self._candidate_revision: int | None = None
        self.next_offset: int | None = None
        self.total_candidates = 0
        self.context: LibraryCharacterRepairContext | None = None
        self.candidates: tuple[CharacterRepairCandidate, ...] = ()
        self.selected_candidate: CharacterRepairCandidate | None = None
        self.status_copy = ""
        self._confirmation_requested = False

    @property
    def historical_identity_copy(self) -> str:
        return self.context.historical_display_snapshot if self.context else ""

    @property
    def identity_comparison(self) -> tuple[str, str] | None:
        if self.context is None or self.selected_candidate is None:
            return None
        candidate = self.selected_candidate
        return (
            self.context.historical_display_snapshot,
            f"{candidate.display_name} · local character {candidate.key.character_id}",
        )

    def accept(
        self, context: LibraryCharacterRepairContext
    ) -> tuple[CharacterRepairCandidate, ...]:
        """Accept one typed context and enumerate only its exact authority."""

        if not isinstance(context, LibraryCharacterRepairContext):
            raise TypeError("context must be a LibraryCharacterRepairContext")
        self.context = context
        self.selected_candidate = None
        self._confirmation_requested = False
        self.status_copy = "Choose a replacement character. Nothing is preselected."
        authority = context.unresolved.data_authority_id
        page = self._service.repair_candidates(context.unresolved, limit=20)
        self.candidates = tuple(
            candidate
            for candidate in page.candidates
            if candidate.key.data_authority_id == authority
        )
        self.next_offset = page.next_offset
        self.total_candidates = page.total
        self._candidate_revision = self._source_revision()
        return self.candidates

    def refresh(
        self, context: LibraryCharacterRepairContext
    ) -> tuple[
        LibraryCharacterRepairContext | None, tuple[CharacterRepairCandidate, ...]
    ]:
        """Reload authoritative evidence/version and same-authority candidates."""

        choices = self.load_refresh(context)
        self.apply_refresh(choices, fallback_context=context)
        return choices.context, self.candidates

    def load_refresh(
        self, context: LibraryCharacterRepairContext, *, offset: int = 0
    ) -> _RepairChoices:
        """Read repair evidence/candidates without mutating presentation state."""

        if not isinstance(context, LibraryCharacterRepairContext):
            raise TypeError("context must be a LibraryCharacterRepairContext")
        revision = self._source_revision()
        restarted = offset > 0 and revision != self._candidate_revision
        if restarted:
            offset = 0
        refresh = getattr(self._service, "refresh_unresolved_evidence", None)
        evidence = (
            refresh(context.unresolved)
            if callable(refresh)
            else (
                context.expected_conversation_version,
                context.historical_display_snapshot,
            )
        )
        if evidence is None:
            return _RepairChoices(None, CharacterRepairPage((), 0, None), revision)
        version, historical = evidence
        refreshed = LibraryCharacterRepairContext(
            unresolved=context.unresolved,
            expected_conversation_version=version,
            historical_display_snapshot=historical,
            return_target=context.return_target,
        )
        page = self._service.repair_candidates(
            refreshed.unresolved, offset=offset, limit=20
        )
        if self._source_revision() != revision:
            # A source write during the read cannot publish an offset continuation.
            return _RepairChoices(None, CharacterRepairPage((), 0, None), None, True)
        return _RepairChoices(refreshed, page, revision, restarted)

    def apply_refresh(
        self,
        choices: _RepairChoices,
        *,
        fallback_context: LibraryCharacterRepairContext,
    ) -> None:
        """Atomically publish a completed refresh on the app thread."""

        self.context = choices.context or fallback_context
        self.candidates = tuple(
            candidate
            for candidate in choices.page.candidates
            if candidate.key.data_authority_id
            == self.context.unresolved.data_authority_id
        )
        self.next_offset = choices.page.next_offset
        self.total_candidates = choices.page.total
        self._candidate_revision = choices.revision
        self.selected_candidate = None
        self._confirmation_requested = False
        self.status_copy = (
            f"{self.total_candidates} local characters. Choose a replacement; nothing is preselected."
            if choices.context is not None
            else "Conversation is no longer available for repair."
        )
        if choices.restarted:
            self.status_copy = (
                "Characters changed. Choices restarted; refresh if needed. "
                + self.status_copy
            )

    def select(self, key: ResolvedLocalCharacterKey) -> bool:
        """Select only an enumerated candidate; display names never select."""

        candidate = next(
            (candidate for candidate in self.candidates if candidate.key == key), None
        )
        self.selected_candidate = candidate
        self._confirmation_requested = False
        return candidate is not None

    def request_confirmation(self) -> bool:
        """Arm the separate confirmation step for the explicit candidate."""

        self._confirmation_requested = self.selected_candidate is not None
        return self._confirmation_requested

    def cancel_confirmation(self) -> None:
        """Cancel confirmation without clearing the recoverable context."""

        self._confirmation_requested = False

    def apply_confirmed(self) -> CharacterRepairResult | None:
        """Commit through Task 2's compare-and-set service after confirmation."""

        admitted = self.prepare_confirmed_repair()
        if admitted is None:
            return None
        request, context = admitted
        result = self.perform_repair(request)
        self.apply_repair_result(result, context)
        return result

    def prepare_confirmed_repair(
        self,
    ) -> tuple[CharacterRepairRequest, LibraryCharacterRepairContext] | None:
        """Consume confirmation and freeze the exact CAS request on the UI thread."""

        context = self.context
        candidate = self.selected_candidate
        if not self._confirmation_requested or context is None or candidate is None:
            return None
        self._confirmation_requested = False
        return (
            CharacterRepairRequest(
                unresolved=context.unresolved,
                replacement=candidate.key,
                expected_conversation_version=context.expected_conversation_version,
            ),
            context,
        )

    def perform_repair(self, request: CharacterRepairRequest) -> CharacterRepairResult:
        """Run the blocking CAS and return a plain typed result only."""

        return self._service.repair(request)

    def apply_repair_result(
        self,
        result: CharacterRepairResult,
        context: LibraryCharacterRepairContext,
    ) -> None:
        """Publish status, invalidation, and navigation on the app thread."""

        if result is CharacterRepairResult.APPLIED:
            self.status_copy = "Character link repaired."
            self._invalidate_keyword()
            self._invalidate_semantic()
            self._return_to_anchor(context.return_target)
            return
        self.status_copy = _RESULT_COPY[result]
        self._focus_refresh()


class LibraryCharacterRepairDialog(ModalScreen[None]):
    """Library-owned explicit selection and two-step repair presentation."""

    DEFAULT_CSS = """
    LibraryCharacterRepairDialog { align: center middle; }
    LibraryCharacterRepairDialog > Container {
        width: 76; max-width: 96%; height: auto; max-height: 90%;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    LibraryCharacterRepairDialog Select { width: 100%; }
    LibraryCharacterRepairDialog Horizontal { height: auto; }
    LibraryCharacterRepairDialog Button { min-width: 14; margin-right: 1; }
    """

    def __init__(
        self,
        controller: LibraryCharacterRepairController,
        context: LibraryCharacterRepairContext,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.context = context
        self._operation_token: object | None = None

    def compose(self) -> ComposeResult:
        with Container(id="library-character-repair-dialog"):
            yield Static("Repair saved character link", classes="dialog-title")
            yield Static(
                f"Historical identity: {self.controller.historical_identity_copy}",
                id="library-character-repair-old-identity",
            )
            yield Select(
                (),
                prompt="Choose a replacement; nothing is preselected",
                allow_blank=True,
                id="library-character-repair-candidate",
            )
            yield Static(
                self.controller.status_copy,
                id="library-character-repair-status",
            )
            yield Button(
                "Next 20 characters", id="library-character-repair-next", disabled=True
            )
            with Horizontal():
                yield Button("Refresh", id="library-character-repair-refresh")
                yield Button("Repair", id="library-character-repair-apply")
                yield Button("Cancel", id="library-character-repair-cancel")

    def on_mount(self) -> None:
        """Load candidates off the UI thread after controls exist."""

        self._start_refresh()

    def _set_busy(self, busy: bool, *, lock_cancel: bool = False) -> None:
        for selector in (
            "#library-character-repair-candidate",
            "#library-character-repair-refresh",
            "#library-character-repair-apply",
            "#library-character-repair-next",
        ):
            self.query_one(selector).disabled = busy
        self.query_one("#library-character-repair-next").disabled = (
            busy or self.controller.next_offset is None
        )
        self.query_one("#library-character-repair-cancel", Button).disabled = (
            busy and lock_cancel
        )

    def _start_refresh(self, *, offset: int = 0) -> None:
        token = object()
        self._operation_token = token
        self._set_busy(True)
        self.query_one("#library-character-repair-status", Static).update(
            "Refreshing authoritative repair choices…"
        )
        self.run_worker(
            self._refresh_owned(token, offset=offset),
            exclusive=False,
            group="library-character-repair-refresh",
        )

    async def _refresh_owned(self, token: object, *, offset: int = 0) -> None:
        try:
            choices = await asyncio.to_thread(
                self.controller.load_refresh, self.context, offset=offset
            )
        except Exception:  # noqa: BLE001 - service boundary becomes retry UI
            if self._operation_token is token and self.is_mounted:
                self.query_one("#library-character-repair-status", Static).update(
                    "Refresh failed. Retry or cancel."
                )
                self._set_busy(False)
            return
        if self._operation_token is not token or not self.is_mounted:
            return
        self.controller.apply_refresh(choices, fallback_context=self.context)
        if choices.context is not None:
            self.context = choices.context
        options = tuple(
            (
                f"{candidate.display_name} · local character {candidate.key.character_id}",
                str(candidate.key.character_id),
            )
            for candidate in self.controller.candidates
        )
        select = self.query_one("#library-character-repair-candidate", Select)
        select.set_options(options)
        select.value = Select.NULL
        self.query_one("#library-character-repair-old-identity", Static).update(
            f"Historical identity: {self.controller.historical_identity_copy}"
        )
        self.query_one("#library-character-repair-status", Static).update(
            self.controller.status_copy
        )
        self.query_one("#library-character-repair-apply", Button).label = "Repair"
        self._operation_token = None
        self._set_busy(False)

    @on(Select.Changed, "#library-character-repair-candidate")
    def _candidate_changed(self, event: Select.Changed) -> None:
        event.stop()
        if event.value is Select.NULL:
            return
        candidate = next(
            (
                item
                for item in self.controller.candidates
                if str(item.key.character_id) == str(event.value)
            ),
            None,
        )
        if candidate is not None and self.controller.select(candidate.key):
            old, selected = self.controller.identity_comparison or ("", "")
            self.query_one("#library-character-repair-status", Static).update(
                f"Replace {old} with {selected}. Press Repair to review."
            )

    @on(Button.Pressed, "#library-character-repair-refresh")
    def _refresh(self, event: Button.Pressed) -> None:
        event.stop()
        self._start_refresh()

    @on(Button.Pressed, "#library-character-repair-next")
    def _next_page(self, event: Button.Pressed) -> None:
        event.stop()
        if self.controller.next_offset is not None:
            self._start_refresh(offset=self.controller.next_offset)

    @on(Button.Pressed, "#library-character-repair-apply")
    def _apply(self, event: Button.Pressed) -> None:
        event.stop()
        button = event.button
        if button.label.plain == "Repair":
            if not self.controller.request_confirmation():
                self.query_one("#library-character-repair-status", Static).update(
                    "Choose a replacement before repairing."
                )
                return
            button.label = "Confirm repair"
            self.query_one("#library-character-repair-status", Static).update(
                "Confirm the old and selected identities before applying."
            )
            return
        token = object()
        admitted = self.controller.prepare_confirmed_repair()
        if admitted is None:
            button.label = "Repair"
            return
        request, context = admitted
        self._operation_token = token
        self._set_busy(True, lock_cancel=True)
        self.run_worker(
            self._apply_owned(token, request, context),
            exclusive=False,
            group="library-character-repair-apply",
        )

    async def _apply_owned(
        self,
        token: object,
        request: CharacterRepairRequest,
        context: LibraryCharacterRepairContext,
    ) -> None:
        try:
            result = await asyncio.to_thread(self.controller.perform_repair, request)
        except Exception:  # noqa: BLE001 - service boundary becomes retry UI
            if self._operation_token is token and self.is_mounted:
                self._operation_token = None
                self.query_one("#library-character-repair-status", Static).update(
                    "Repair failed. Retry or cancel."
                )
                self.query_one(
                    "#library-character-repair-apply", Button
                ).label = "Repair"
                self._set_busy(False)
            return
        if self._operation_token is not token or not self.is_mounted:
            return
        self.controller.apply_repair_result(result, context)
        self._operation_token = None
        self.query_one("#library-character-repair-status", Static).update(
            self.controller.status_copy
        )
        if result is CharacterRepairResult.APPLIED:
            self.dismiss(None)
        else:
            self.query_one("#library-character-repair-apply", Button).label = "Repair"
            self._set_busy(False)
            refresh = self.query_one("#library-character-repair-refresh", Button)
            # The pressed Apply button regains focus as its click dispatch
            # settles; restore the required recovery anchor afterward.
            self.set_timer(0.01, refresh.focus)

    @on(Button.Pressed, "#library-character-repair-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.controller.cancel_confirmation()
        self.dismiss(None)
