"""Console per-conversation Library access modal tests."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable

import pytest
from textual.widgets import Button, RadioButton, RadioSet, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_GLYPH_RADIO_SELECTED,
    LIBRARY_GLYPH_RADIO_UNSELECTED,
    LIBRARY_GLYPH_SELECTED,
    LIBRARY_GLYPH_UNSELECTED,
)
from tldw_chatbook.Chat.console_display_state import (
    ConsoleLibraryPolicyDisplayState,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Widgets.Console import console_library_access_modal
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


class _ModalApp(ConsolidatedCSSApp):
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
        assert modal.focused is modal.query_one("#library-auto-policy", RadioSet)


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
        # TASK-25824: once Discard appears, "Cancel" is ambiguous -- it no
        # longer means "abandon my edits" (that is Discard) but "stay here".
        # The pair must name the two outcomes it actually offers.
        assert (
            str(modal.query_one("#library-access-cancel", Button).label)
            == "Keep editing"
        )

        # Qodo review (PR #2256): reverting to the saved values makes the modal
        # clean again, and a clean Cancel dismisses immediately -- so the
        # "Keep editing" label must not survive the state that justified it.
        await pilot.click("#library-auto-never")
        await pilot.pause()
        assert modal.query_one("#library-access-discard", Button).display is False
        assert (
            str(modal.query_one("#library-access-cancel", Button).label) == "Cancel"
        )


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


@pytest.mark.asyncio
async def test_save_failure_preserves_choices_and_focuses_retryable_feedback() -> None:
    snapshot = _snapshot()

    async def fail_save(
        _candidate: ConsoleLibraryPolicyCandidate,
    ) -> ConsoleLibraryPolicySaveOutcome:
        raise RuntimeError("write failed")

    modal = ConsoleLibraryAccessModal(
        snapshot=snapshot,
        state=_state(snapshot),
        save_policy=fail_save,
        reload_policy=lambda: _async_snapshot(snapshot),
    )
    app = _ModalApp(modal)

    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        await pilot.click("#library-agent-allowed")
        await pilot.click("#library-access-save")
        await pilot.pause()

        feedback = modal.query_one("#library-access-feedback", Static)
        assert modal.query_one("#library-agent-allowed", RadioButton).value is True
        assert modal.query_one("#library-access-save", Button).disabled is False
        assert feedback.can_focus is True
        assert modal.focused is feedback
        assert "unsaved choices are still here" in str(feedback.renderable)


@pytest.mark.asyncio
async def test_compare_retry_is_single_flight_while_reload_is_pending() -> None:
    snapshot = _snapshot()
    entered = asyncio.Event()
    release = asyncio.Event()
    reload_calls = 0

    async def delayed_reload() -> ConsoleLibraryPolicySnapshot:
        nonlocal reload_calls
        reload_calls += 1
        entered.set()
        await release.wait()
        return snapshot

    modal = ConsoleLibraryAccessModal(
        snapshot=snapshot,
        state=_state(snapshot),
        save_policy=_saved,
        reload_policy=delayed_reload,
    )
    app = _ModalApp(modal)

    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        await pilot.click("#library-agent-allowed")
        first = asyncio.create_task(modal._reload(preserve_candidate=True))
        await entered.wait()

        await modal._reload(preserve_candidate=True)
        assert reload_calls == 1

        release.set()
        await first
        assert reload_calls == 1


@pytest.mark.parametrize("dismissal", ("escape", "backdrop"))
@pytest.mark.asyncio
async def test_clean_dismissal_restores_the_library_access_opener(
    dismissal: str,
) -> None:
    snapshot = _snapshot()
    modal = ConsoleLibraryAccessModal(
        snapshot=snapshot,
        state=_state(snapshot),
        save_policy=_saved,
        reload_policy=lambda: _async_snapshot(snapshot),
    )

    class OpenerApp(ConsolidatedCSSApp):
        def compose(self):
            yield Button("Library", id="library-access-opener")

        def on_mount(self) -> None:
            self.query_one("#library-access-opener", Button).focus()
            self.push_screen(modal)

    app = OpenerApp()
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        if dismissal == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()
        await pilot.pause()

        assert app.screen is not modal
        assert app.screen.focused is app.query_one("#library-access-opener", Button)


@pytest.mark.asyncio
async def test_the_access_radios_paint_the_shared_radio_glyph_pair() -> None:
    """task-32464 AC#3: the painted radio glyphs come from the legend.

    The Library legend gives "○" one meaning -- blocked or disabled -- and
    this modal painted it on an unselected radio two sentences away from the
    Console guide's ☑/☐ paragraph. The controller's ruling keeps ●/○ here (a
    radio is not a checkbox) and carves radios out of the legend explicitly,
    so what this surface paints is now pinned BOTH ways: it wears the radio
    pair, and it must not drift onto the multi-select pair.
    """
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
        chosen = modal.query_one("#library-auto-never", RadioButton)
        unchosen = modal.query_one("#library-auto-automatic", RadioButton)
        assert (chosen.value, unchosen.value) == (True, False)

        painted_chosen = chosen.render().plain
        painted_unchosen = unchosen.render().plain
        assert LIBRARY_GLYPH_RADIO_SELECTED in painted_chosen, painted_chosen
        assert LIBRARY_GLYPH_RADIO_UNSELECTED in painted_unchosen, painted_unchosen
        # A chooser must not promise the multi-select it does not offer.
        assert LIBRARY_GLYPH_SELECTED not in painted_chosen, painted_chosen
        assert LIBRARY_GLYPH_UNSELECTED not in painted_unchosen, painted_unchosen

    assert (LIBRARY_GLYPH_RADIO_SELECTED, LIBRARY_GLYPH_RADIO_UNSELECTED) == (
        "●",
        "○",
    )


def test_the_access_radio_glyphs_are_not_local_literals() -> None:
    """task-32464 AC#2: one source of truth, not a third set of literals.

    Value equality cannot catch this on its own -- the ruling keeps the same
    two characters -- so the pin reads the module's own source. Without it a
    future legend change would edit the constants and silently pass this
    surface by, which is exactly how this defect survived task-32303.
    """
    source = inspect.getsource(console_library_access_modal)
    assert '"●"' not in source, "the selected radio glyph is a local literal"
    assert '"○"' not in source, "the unselected radio glyph is a local literal"
