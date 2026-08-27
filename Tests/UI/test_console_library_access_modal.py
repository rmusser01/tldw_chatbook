"""Console per-conversation Library access modal tests."""

from __future__ import annotations

from collections.abc import Awaitable

import pytest
from textual.app import App
from textual.widgets import Button, RadioButton, Static

from tldw_chatbook.Chat.console_display_state import (
    ConsoleLibraryPolicyDisplayState,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Widgets.Console.console_library_access_modal import (
    ConsoleLibraryAccessModal,
    ConsoleLibraryPolicySaveOutcome,
)


def _snapshot(*, source: str = "durable") -> ConsoleLibraryPolicySnapshot:
    return ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        policy_revision=3 if source == "durable" else None,
        source=source,  # type: ignore[arg-type]
    )


def _state(snapshot: ConsoleLibraryPolicySnapshot, **overrides):
    return ConsoleLibraryPolicyDisplayState.from_snapshot(snapshot, **overrides)


class _ModalApp(App[None]):
    def __init__(
        self,
        modal: ConsoleLibraryAccessModal,
    ) -> None:
        super().__init__()
        self.modal = modal

    def on_mount(self) -> None:
        self.push_screen(self.modal)


def _saved(
    candidate: ConsoleLibraryPolicyCandidate,
) -> Awaitable[ConsoleLibraryPolicySaveOutcome]:
    async def finish() -> ConsoleLibraryPolicySaveOutcome:
        return ConsoleLibraryPolicySaveOutcome(
            status="saved",
            snapshot=ConsoleLibraryPolicySnapshot(
                auto_retrieve=candidate.auto_retrieve,
                assistant_access=candidate.assistant_access,
                policy_revision=4,
                source="durable",
            ),
            copy="Saved on this device. This policy is not synced.",
        )

    return finish()


@pytest.mark.asyncio
async def test_access_modal_separates_the_two_text_valued_axes_and_disclosures() -> None:
    snapshot = _snapshot()
    modal = ConsoleLibraryAccessModal(
        snapshot=snapshot,
        state=_state(
            snapshot,
            provider_intent_label="Library tool mode: RAG",
            resolved_destination_label="Resolved destination: public network",
        ),
        save_policy=_saved,
        reload_policy=lambda: _async_snapshot(snapshot),
    )
    app = _ModalApp(modal)

    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        assert modal.query_one("#library-auto-never", RadioButton).value is True
        assert modal.query_one("#library-agent-blocked", RadioButton).value is True
        text = " ".join(str(item.renderable) for item in modal.query(Static))
        assert "Stored only on this device" in text
        assert "not synced" in text
        assert "Library tool mode: RAG" in text
        assert "Resolved destination: public network" in text
        assert modal.query_one("#library-access-save", Button).disabled is True


async def _async_snapshot(
    snapshot: ConsoleLibraryPolicySnapshot,
) -> ConsoleLibraryPolicySnapshot:
    return snapshot


@pytest.mark.asyncio
async def test_dirty_escape_requires_explicit_discard_and_preserves_the_edit() -> None:
    snapshot = _snapshot()
    modal = ConsoleLibraryAccessModal(
        snapshot=snapshot,
        state=_state(snapshot),
        save_policy=_saved,
        reload_policy=lambda: _async_snapshot(snapshot),
    )
    app = _ModalApp(modal)

    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        await pilot.click("#library-auto-automatic")
        await pilot.pause()
        assert modal.query_one("#library-access-save", Button).disabled is False

        await pilot.press("escape")
        await pilot.pause()

        assert app.screen is modal
        assert modal.query_one("#library-auto-automatic", RadioButton).value is True
        assert modal.query_one("#library-access-discard", Button).display is True
        feedback = modal.query_one("#library-access-feedback", Static)
        assert "Unsaved changes" in str(feedback.renderable)


@pytest.mark.asyncio
async def test_conflict_is_persistent_and_exposes_reload_and_compare_retry() -> None:
    snapshot = _snapshot()

    async def conflict(
        _candidate: ConsoleLibraryPolicyCandidate,
    ) -> ConsoleLibraryPolicySaveOutcome:
        return ConsoleLibraryPolicySaveOutcome(
            status="conflict",
            snapshot=snapshot,
            copy="This policy changed elsewhere. Reload or compare and retry.",
        )

    modal = ConsoleLibraryAccessModal(
        snapshot=snapshot,
        state=_state(snapshot),
        save_policy=conflict,
        reload_policy=lambda: _async_snapshot(snapshot),
    )
    app = _ModalApp(modal)

    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        await pilot.click("#library-agent-allowed")
        await pilot.click("#library-access-save")
        await pilot.pause()

        assert app.screen is modal
        feedback = modal.query_one("#library-access-feedback", Static)
        assert "changed elsewhere" in str(feedback.renderable)
        assert modal.query_one("#library-access-reload", Button).display is True
        assert modal.query_one("#library-access-compare-retry", Button).display is True
        assert modal.query_one("#library-agent-allowed", RadioButton).value is True


@pytest.mark.asyncio
async def test_unavailable_policy_is_fail_closed_and_not_editable() -> None:
    snapshot = ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        policy_revision=None,
        source="unavailable",
        error_code="policy_read_error",
    )
    modal = ConsoleLibraryAccessModal(
        snapshot=snapshot,
        state=_state(snapshot),
        save_policy=_saved,
        reload_policy=lambda: _async_snapshot(snapshot),
    )
    app = _ModalApp(modal)

    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        assert modal.query_one("#library-auto-never", RadioButton).disabled is True
        assert modal.query_one("#library-agent-blocked", RadioButton).disabled is True
        assert modal.query_one("#library-access-save", Button).disabled is True
        assert "Unavailable" in str(
            modal.query_one("#library-access-status", Static).renderable
        )
