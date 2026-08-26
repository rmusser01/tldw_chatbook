from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Chat.console_prompt_queue import (
    MAX_CONSOLE_QUEUE_ENTRIES,
    ConsolePromptQueueRegistry,
    PromptQueueMutationResult,
    PromptQueuePauseReason,
    QueueMutationStatus,
)
from tldw_chatbook.Widgets.Console.console_prompt_queue_modal import (
    ConsolePromptQueueModal,
)


class _QueueFacade:
    def __init__(
        self, *, pause_reason: PromptQueuePauseReason | None = None
    ) -> None:
        self.registry = ConsolePromptQueueRegistry()
        snapshot = self.registry.snapshot("pinned-session")
        snapshot = self.registry.begin_chain(
            "pinned-session", context_epoch=1, expected_revision=snapshot.revision
        ).snapshot
        for text in ("first [safe] prompt", "second prompt"):
            snapshot = self.registry.admit(
                "pinned-session", text=text, expected_revision=snapshot.revision
            ).snapshot
        if pause_reason is not None:
            self.registry.pause(
                "pinned-session",
                reason=pause_reason,
                expected_revision=snapshot.revision,
            )
        self.read_calls: list[tuple[str, str, int]] = []
        self.recover_calls: list[tuple[str, str, int, int | None]] = []

    def snapshot(self, session_id: str):
        return self.registry.snapshot(session_id)

    def read_waiting_text(self, session_id: str, entry_id: str, *, expected_revision: int):
        self.read_calls.append((session_id, entry_id, expected_revision))
        return self.registry.read_waiting_text(
            session_id,
            entry_id=entry_id,
            expected_revision=expected_revision,
        )

    def edit_waiting(self, session_id: str, entry_id: str, *, text: str, expected_revision: int):
        return self.registry.edit(
            session_id,
            entry_id=entry_id,
            text=text,
            expected_revision=expected_revision,
        )

    def move_waiting(self, session_id: str, entry_id: str, *, position: int, expected_revision: int):
        return self.registry.move(
            session_id,
            entry_id=entry_id,
            new_index=position,
            expected_revision=expected_revision,
        )

    def remove_waiting(self, session_id: str, entry_id: str, *, expected_revision: int):
        return self.registry.remove(
            session_id,
            entry_id=entry_id,
            expected_revision=expected_revision,
        )

    def clear_waiting(self, session_id: str, *, expected_revision: int):
        return self.registry.clear_waiting(
            session_id, expected_revision=expected_revision
        )

    async def toggle_pause(self, session_id: str, *, expected_revision: int):
        return self.registry.request_pause_after_turn(
            session_id, expected_revision=expected_revision
        )

    def context_review(self, session_id: str) -> tuple[int | None, int]:
        assert session_id == "pinned-session"
        return (1, 7)

    async def recover(
        self,
        session_id: str,
        *,
        action: str,
        expected_revision: int,
        reviewed_context_epoch: int | None = None,
    ) -> PromptQueueMutationResult:
        self.recover_calls.append(
            (session_id, action, expected_revision, reviewed_context_epoch)
        )
        snapshot = self.registry.snapshot(session_id)
        if action == "use-current-context" and reviewed_context_epoch is None:
            return PromptQueueMutationResult(
                QueueMutationStatus.INVALID,
                snapshot,
                detail="Review the current context before using it.",
            )
        return PromptQueueMutationResult(QueueMutationStatus.UNCHANGED, snapshot)


@pytest.mark.asyncio
async def test_manager_fetches_no_body_until_selected_edit_begins() -> None:
    facade = _QueueFacade()
    snapshot = facade.snapshot("pinned-session")
    app = App()

    async with app.run_test(size=(80, 24)) as pilot:
        modal = ConsolePromptQueueModal(
            session_id="pinned-session",
            revision=snapshot.revision,
            queue_controller=facade,
        )
        app.push_screen(modal)
        await pilot.pause()

        assert facade.read_calls == []
        assert modal.session_id == "pinned-session"
        state = modal.query_one("#console-prompt-queue-manager-state", Static)
        assert f"/{MAX_CONSOLE_QUEUE_ENTRIES}" in str(state.renderable)

        await pilot.click("#console-prompt-queue-edit")
        await pilot.pause()

        assert len(facade.read_calls) == 1
        assert facade.read_calls[0][0] == "pinned-session"
        assert modal.query_one("#console-prompt-queue-edit-input").text == (
            "first [safe] prompt"
        )


@pytest.mark.asyncio
async def test_manager_rejects_unsafe_edited_prompt_at_ui_boundary() -> None:
    facade = _QueueFacade()
    before = facade.snapshot("pinned-session")
    first_entry = before.entries[0]
    app = App()

    async with app.run_test(size=(80, 24)) as pilot:
        modal = ConsolePromptQueueModal(
            session_id="pinned-session",
            revision=before.revision,
            queue_controller=facade,
        )
        app.push_screen(modal)
        await pilot.pause()

        await pilot.click("#console-prompt-queue-edit")
        editor = modal.query_one("#console-prompt-queue-edit-input", TextArea)
        editor.text = "<script>alert('queued')</script>"
        await pilot.click("#console-prompt-queue-save")
        await pilot.pause()

        after = facade.snapshot("pinned-session")
        assert after.revision == before.revision
        assert after.entries[0] is first_entry
        feedback = modal.query_one(
            "#console-prompt-queue-manager-feedback", Static
        )
        assert "Prompt blocked" in str(feedback.renderable)


@pytest.mark.asyncio
async def test_manager_keeps_pinned_session_and_recovers_from_stale_revision() -> None:
    facade = _QueueFacade()
    snapshot = facade.snapshot("pinned-session")
    app = App()

    async with app.run_test(size=(160, 40)) as pilot:
        modal = ConsolePromptQueueModal(
            session_id="pinned-session",
            revision=snapshot.revision,
            queue_controller=facade,
        )
        app.push_screen(modal)
        await pilot.pause()

        facade.registry.admit(
            "pinned-session",
            text="external change",
            expected_revision=snapshot.revision,
        )
        modal._move_selected(1)
        await pilot.pause()

        assert modal.session_id == "pinned-session"
        assert modal._revision == facade.snapshot("pinned-session").revision
        feedback = modal.query_one("#console-prompt-queue-manager-feedback")
        assert "Queue changed" in str(feedback.renderable)


@pytest.mark.asyncio
async def test_use_current_context_requires_and_reuses_explicit_review_epoch() -> None:
    facade = _QueueFacade(pause_reason=PromptQueuePauseReason.CONTEXT_CHANGED)
    snapshot = facade.snapshot("pinned-session")
    app = App()

    async with app.run_test(size=(100, 30)) as pilot:
        modal = ConsolePromptQueueModal(
            session_id="pinned-session",
            revision=snapshot.revision,
            queue_controller=facade,
        )
        app.push_screen(modal)
        await pilot.pause()

        use_current = modal.query_one("#console-prompt-queue-use-context", Button)
        assert use_current.disabled
        assert facade.recover_calls == []

        await pilot.click("#console-prompt-queue-review-context")
        assert not use_current.disabled
        await pilot.click("#console-prompt-queue-use-context")
        await pilot.pause()
        assert facade.recover_calls[-1] == (
            "pinned-session",
            "use-current-context",
            snapshot.revision,
            7,
        )


@pytest.mark.asyncio
async def test_remove_and_clear_require_explicit_destructive_confirmation() -> None:
    facade = _QueueFacade()
    snapshot = facade.snapshot("pinned-session")
    app = App()

    async with app.run_test(size=(100, 30)) as pilot:
        modal = ConsolePromptQueueModal(
            session_id="pinned-session",
            revision=snapshot.revision,
            queue_controller=facade,
        )
        app.push_screen(modal)
        await pilot.pause()

        await pilot.click("#console-prompt-queue-remove")
        await pilot.click("#continue-btn")
        await pilot.pause()
        assert facade.snapshot("pinned-session").total_count == 2

        await pilot.click("#console-prompt-queue-remove")
        await pilot.click("#cancel-btn")
        await pilot.pause()
        assert facade.snapshot("pinned-session").total_count == 1

        # Textual buttons deliberately debounce rapid repeated activations.
        await pilot.pause(0.3)
        await pilot.click("#console-prompt-queue-clear")
        await pilot.click("#cancel-btn")
        await pilot.pause()
        assert facade.snapshot("pinned-session").total_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (160, 40)])
async def test_manager_controls_remain_inside_dialog_at_supported_sizes(size) -> None:
    facade = _QueueFacade()
    snapshot = facade.snapshot("pinned-session")
    app = App()

    async with app.run_test(size=size) as pilot:
        modal = ConsolePromptQueueModal(
            session_id="pinned-session",
            revision=snapshot.revision,
            queue_controller=facade,
        )
        app.push_screen(modal)
        await pilot.pause()
        dialog = modal.query_one("#console-prompt-queue-dialog")

        for button in dialog.query(Button):
            assert button.region.x >= dialog.region.x
            assert button.region.y >= dialog.region.y
            assert button.region.right <= dialog.region.right
            assert button.region.bottom <= dialog.region.bottom
