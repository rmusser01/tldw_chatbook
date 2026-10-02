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
from tldw_chatbook.UI.Console_Modules.prompt_queue import ConsoleQueueRecoveryTurn
from tldw_chatbook.Widgets.Console.console_prompt_queue_modal import (
    ConsolePromptQueueModal,
)


class _QueueFacade:
    def __init__(
        self,
        *,
        pause_reason: PromptQueuePauseReason | None = None,
        recovery_turns: dict[str, ConsoleQueueRecoveryTurn] | None = None,
    ) -> None:
        self.recovery_turns = dict(recovery_turns or {})
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

    def recovery_turn(self, session_id: str, *, action: str):
        assert session_id == "pinned-session"
        return self.recovery_turns.get(action)

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
        await pilot.click("#cancel-button")
        await pilot.pause()
        assert facade.snapshot("pinned-session").total_count == 2

        await pilot.click("#console-prompt-queue-remove")
        await pilot.click("#confirm-button")
        await pilot.pause()
        assert facade.snapshot("pinned-session").total_count == 1

        # Textual buttons deliberately debounce rapid repeated activations.
        await pilot.pause(0.3)
        await pilot.click("#console-prompt-queue-clear")
        await pilot.click("#confirm-button")
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
            if not _shown(button):
                continue  # TASK-33621.19: inapplicable actions are hidden
            assert button.region.x >= dialog.region.x
            assert button.region.y >= dialog.region.y
            assert button.region.right <= dialog.region.right
            assert button.region.bottom <= dialog.region.bottom


# ---------------------------------------------------------------------------
# TASK-33621.19 AC#5 (GAP1-04 / GAP5-09): the manager numbered rows from 0,
# used a disabled "Paused" button as its status, showed 13 buttons with 7
# disabled and unexplained, and labelled resume-next "Skip & resume" although
# it skips (discards) nothing.
# ---------------------------------------------------------------------------


def _shown(widget) -> bool:
    node = widget
    while node is not None and not isinstance(node, ConsolePromptQueueModal):
        if not node.display:
            return False
        node = node.parent
    return True


def _labels(modal: ConsolePromptQueueModal) -> dict[str, str]:
    return {
        button.id: str(button.label)
        for button in modal.query(Button)
        if button.id and _shown(button)
    }


async def _open(app: App, pilot, facade: _QueueFacade) -> ConsolePromptQueueModal:
    modal = ConsolePromptQueueModal(
        session_id="pinned-session",
        revision=facade.snapshot("pinned-session").revision,
        queue_controller=facade,
    )
    app.push_screen(modal)
    await pilot.pause()
    return modal


def _state_text(modal: ConsolePromptQueueModal) -> str:
    return str(
        modal.query_one("#console-prompt-queue-manager-state", Static).renderable
    )


_NEVER_LABELS = {"Paused", "Skip & resume"}
_RECOVERY_ONLY = {
    "console-prompt-queue-resume-next",
    "console-prompt-queue-review-context",
    "console-prompt-queue-use-context",
    "console-prompt-queue-retry-failed",
    "console-prompt-queue-retry-stopped",
}


@pytest.mark.asyncio
async def test_manager_numbers_from_one_and_hides_actions_that_do_not_apply() -> None:
    facade = _QueueFacade()
    app = App()

    async with app.run_test(size=(100, 30)) as pilot:
        modal = await _open(app, pilot, facade)

        rows = [
            str(button.label)
            for button in modal.query(".console-prompt-queue-entry-select")
        ]
        assert rows == ["1. first [safe] prompt", "2. second prompt"]
        assert _state_text(modal) == f"Queue 2/{MAX_CONSOLE_QUEUE_ENTRIES} · Draining"

        labels = _labels(modal)
        assert not _RECOVERY_ONLY & labels.keys()
        assert "console-prompt-queue-save" not in labels  # edit mode only
        assert labels["console-prompt-queue-toggle-pause"] == "Pause"
        assert not _NEVER_LABELS & set(labels.values())
        for button in modal.query(Button):
            if _shown(button) and button.disabled:
                assert button.tooltip, f"{button.id} is disabled with no reason"

        await pilot.click("#console-prompt-queue-edit")
        await pilot.pause()
        assert "console-prompt-queue-save" in _labels(modal)


@pytest.mark.asyncio
async def test_manager_names_a_real_failed_turn_and_offers_retry_and_resume_next() -> (
    None
):
    facade = _QueueFacade(
        pause_reason=PromptQueuePauseReason.FAILED,
        recovery_turns={
            "retry-failed": ConsoleQueueRecoveryTurn(
                "assistant-2", "Answer with the word BRAVO."
            )
        },
    )
    app = App()

    async with app.run_test(size=(100, 30)) as pilot:
        modal = await _open(app, pilot, facade)

        assert _state_text(modal) == (
            f"Queue 2/{MAX_CONSOLE_QUEUE_ENTRIES} · Paused · "
            'Turn failed: "Answer with the word BRAVO."'
        )
        labels = _labels(modal)
        # The state is text; no disabled "Paused" button stands in for it.
        assert "console-prompt-queue-toggle-pause" not in labels
        assert labels["console-prompt-queue-retry-failed"] == "Retry failed"
        assert labels["console-prompt-queue-resume-next"] == "Resume next"
        assert "console-prompt-queue-retry-stopped" not in labels
        assert "console-prompt-queue-review-context" not in labels
        assert not _NEVER_LABELS & set(labels.values())
        assert not modal.query_one(
            "#console-prompt-queue-retry-failed", Button
        ).disabled

        await pilot.click("#console-prompt-queue-retry-failed")
        await pilot.pause()
        assert facade.recover_calls[-1][1] == "retry-failed"


@pytest.mark.asyncio
async def test_manager_drops_retry_when_the_failed_turn_changes_while_open() -> None:
    """The retry target is transcript state the queue revision never bumps.

    While the manager is open, the named failed turn can stop being the
    newest failed reply. Its "Retry failed" must not linger to refuse with
    "No matching stopped or failed turn is available." (TASK-33621.19 AC#3).
    """

    facade = _QueueFacade(
        pause_reason=PromptQueuePauseReason.FAILED,
        recovery_turns={
            "retry-failed": ConsoleQueueRecoveryTurn(
                "assistant-2", "Answer with the word BRAVO."
            )
        },
    )
    revision = facade.snapshot("pinned-session").revision
    app = App()

    async with app.run_test(size=(100, 30)) as pilot:
        modal = await _open(app, pilot, facade)
        assert "console-prompt-queue-retry-failed" in _labels(modal)

        facade.recovery_turns.clear()
        await pilot.pause(0.5)

        assert facade.snapshot("pinned-session").revision == revision
        assert _state_text(modal) == f"Queue 2/{MAX_CONSOLE_QUEUE_ENTRIES} · Paused"
        labels = _labels(modal)
        assert "console-prompt-queue-retry-failed" not in labels
        assert labels["console-prompt-queue-toggle-pause"] == "Resume"


@pytest.mark.asyncio
async def test_manager_failed_pause_without_a_failed_turn_offers_resume() -> None:
    facade = _QueueFacade(pause_reason=PromptQueuePauseReason.FAILED)
    app = App()

    async with app.run_test(size=(100, 30)) as pilot:
        modal = await _open(app, pilot, facade)

        state = _state_text(modal)
        assert state == f"Queue 2/{MAX_CONSOLE_QUEUE_ENTRIES} · Paused"
        assert "failed" not in state.lower()
        labels = _labels(modal)
        assert labels["console-prompt-queue-toggle-pause"] == "Resume"
        assert not modal.query_one(
            "#console-prompt-queue-toggle-pause", Button
        ).disabled
        assert not _RECOVERY_ONLY & labels.keys()


@pytest.mark.asyncio
async def test_manager_stopped_pause_offers_resume_next_and_retry_stopped() -> None:
    facade = _QueueFacade(
        pause_reason=PromptQueuePauseReason.STOPPED,
        recovery_turns={
            "retry-stopped": ConsoleQueueRecoveryTurn("assistant-1", "ALPHA")
        },
    )
    app = App()

    async with app.run_test(size=(100, 30)) as pilot:
        modal = await _open(app, pilot, facade)

        assert _state_text(modal) == (
            f"Queue 2/{MAX_CONSOLE_QUEUE_ENTRIES} · Paused · Turn stopped"
        )
        labels = _labels(modal)
        assert "console-prompt-queue-toggle-pause" not in labels
        assert labels["console-prompt-queue-resume-next"] == "Resume next"
        assert labels["console-prompt-queue-retry-stopped"] == "Retry stopped"
        assert "console-prompt-queue-retry-failed" not in labels


@pytest.mark.asyncio
async def test_manager_mutation_feedback_after_dismissal_is_a_no_op() -> None:
    facade = _QueueFacade()
    app = App()

    async with app.run_test(size=(100, 30)) as pilot:
        modal = await _open(app, pilot, facade)
        app.pop_screen()
        await pilot.pause()
        # Textual never resets ``is_mounted``; detachment is the real signal.
        assert not modal.is_attached

        # A queue action that finishes after the manager closed must not
        # query the dismissed modal's widgets.
        modal._accept_mutation(
            PromptQueueMutationResult(
                QueueMutationStatus.STALE_REVISION,
                facade.snapshot("pinned-session"),
            )
        )


@pytest.mark.asyncio
async def test_closing_the_manager_does_not_cancel_a_resume_it_started() -> None:
    """Resume drains whole turns; closing the manager must not cancel them.

    Live (TASK-33621.19 verification): pressing Resume in Manage and then
    closing it cancelled the queued turn in flight ("Response failed"),
    because the drain ran in a worker the modal owned.
    """

    import asyncio

    facade = _QueueFacade(pause_reason=PromptQueuePauseReason.FAILED)
    release = asyncio.Event()
    outcome: list[str] = []

    async def toggle_pause(session_id: str, *, expected_revision: int):
        try:
            await release.wait()
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise
        outcome.append("drained")
        return PromptQueueMutationResult(
            QueueMutationStatus.APPLIED, facade.snapshot(session_id)
        )

    facade.toggle_pause = toggle_pause
    app = App()

    async with app.run_test(size=(100, 30)) as pilot:
        modal = await _open(app, pilot, facade)
        assert _labels(modal)["console-prompt-queue-toggle-pause"] == "Resume"
        await pilot.click("#console-prompt-queue-toggle-pause")
        await pilot.pause()

        app.pop_screen()
        await pilot.pause()
        assert not modal.is_attached
        release.set()
        await pilot.pause()
        await pilot.pause()

    assert outcome == ["drained"]


@pytest.mark.asyncio
async def test_closing_the_manager_does_not_cancel_a_retry_it_started() -> None:
    """The Retry/Resume-next twin: recovery drains whole turns as well."""

    import asyncio

    facade = _QueueFacade(
        pause_reason=PromptQueuePauseReason.FAILED,
        recovery_turns={
            "retry-failed": ConsoleQueueRecoveryTurn(
                "assistant-2", "Answer with the word BRAVO."
            )
        },
    )
    release = asyncio.Event()
    outcome: list[str] = []

    async def recover(
        session_id: str,
        *,
        action: str,
        expected_revision: int,
        reviewed_context_epoch: int | None = None,
    ):
        try:
            await release.wait()
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise
        outcome.append(action)
        return PromptQueueMutationResult(
            QueueMutationStatus.APPLIED, facade.snapshot(session_id)
        )

    facade.recover = recover
    app = App()

    async with app.run_test(size=(100, 30)) as pilot:
        modal = await _open(app, pilot, facade)
        await pilot.click("#console-prompt-queue-retry-failed")
        await pilot.pause()

        app.pop_screen()
        await pilot.pause()
        assert not modal.is_attached
        release.set()
        await pilot.pause()
        await pilot.pause()

    assert outcome == ["retry-failed"]
