"""Joined production-UI workflows for Console Library controls."""

from __future__ import annotations

import pytest
from textual import events
from textual.app import ComposeResult
from textual.keys import _character_to_key
from textual.widgets import RadioButton, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_display_state import (
    ConsoleControlState,
    ConsoleLibraryPolicyDisplayState,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.UI.Workbench import WorkbenchActionRequested
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_library_access_modal import (
    ConsoleLibraryAccessModal,
    ConsoleLibraryPolicySaveOutcome,
)
from tldw_chatbook.Widgets.Console.console_library_search_modal import (
    ConsoleLibrarySearchModal,
)
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleLibraryChip,
    ConsoleStatusChips,
)


def _snapshot(
    auto: ConsoleAutoRetrieve,
    assistant: ConsoleAssistantLibraryAccess,
) -> ConsoleLibraryPolicySnapshot:
    return ConsoleLibraryPolicySnapshot(auto, assistant, 3, "durable")


class _ChipApp(ConsolidatedCSSApp):
    def __init__(self, snapshot: ConsoleLibraryPolicySnapshot) -> None:
        super().__init__()
        self.snapshot = snapshot

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(
            ConsoleControlState.from_values(library_policy=self.snapshot)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auto", "assistant", "label"),
    (
        (
            ConsoleAutoRetrieve.NEVER,
            ConsoleAssistantLibraryAccess.BLOCKED,
            "Library · Auto off · Agent access off",
        ),
        (
            ConsoleAutoRetrieve.NEVER,
            ConsoleAssistantLibraryAccess.ALLOWED,
            "Library · Auto off · Agent access on",
        ),
        (
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.BLOCKED,
            "Library · Auto on · Agent access off",
        ),
        (
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.ALLOWED,
            "Library · Auto on · Agent access on",
        ),
    ),
)
async def test_four_policy_states_render_exactly_and_open_independent_axes(
    auto: ConsoleAutoRetrieve,
    assistant: ConsoleAssistantLibraryAccess,
    label: str,
) -> None:
    snapshot = _snapshot(auto, assistant)
    app = _ChipApp(snapshot)

    async with app.run_test(size=(120, 35)) as pilot:
        chip = app.query_one("#console-library-chip")
        assert str(chip.render()) == label

        async def save(candidate) -> ConsoleLibraryPolicySaveOutcome:
            return ConsoleLibraryPolicySaveOutcome("saved", snapshot, "Saved.")

        async def reload() -> ConsoleLibraryPolicySnapshot:
            return snapshot

        modal = ConsoleLibraryAccessModal(
            snapshot=snapshot,
            state=ConsoleLibraryPolicyDisplayState.from_snapshot(
                snapshot,
                provider_intent_label="Library tool mode: RAG",
                resolved_destination_label="Resolved destination: public network",
            ),
            save_policy=save,
            reload_policy=reload,
        )
        await app.push_screen(modal)
        await pilot.pause()

        assert modal.query_one("#library-auto-never", RadioButton).value is (
            auto is ConsoleAutoRetrieve.NEVER
        )
        assert modal.query_one("#library-agent-blocked", RadioButton).value is (
            assistant is ConsoleAssistantLibraryAccess.BLOCKED
        )
        copy = " ".join(str(row.renderable) for row in modal.query(Static))
        assert "Stored only on this device" in copy
        assert "Library tool mode: RAG" in copy
        assert "Resolved destination: public network" in copy


@pytest.mark.asyncio
async def test_manual_search_is_available_with_safe_defaults_and_preserves_draft() -> (
    None
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    draft = "  exact manual query 研究🙂  "

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-command-input")
        assert ConsoleControlState.from_values().rag_label == (
            "Library · Auto off · Agent access off"
        )
        console.query_one(ConsoleComposerBar).load_draft(draft)

        console.post_message(WorkbenchActionRequested("run-library-rag"))
        await pilot.pause()
        await pilot.pause()

        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleLibrarySearchModal)
        assert modal._query == draft


async def _focused_library_chip(host, pilot):
    """Mount the real Console and focus its Library chip, as a dismissal does.

    Dismissing the Library access dialog hands focus back to the chip that
    opened it (TASK-16211), so the next thing a user types lands on the chip.
    """
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-library-chip")
    chip = console.query_one("#console-library-chip", ConsoleLibraryChip)
    chip.focus()
    await pilot.pause()
    assert host.focused is chip
    return console


def _send_key_burst(host, keys: str) -> None:
    """Deliver ``keys`` the way a terminal delivers typed text: one burst.

    Every key is queued before any is handled, so each one reaches the
    focused chip before the dialog its first activation opens is on screen.
    ``pilot.press`` cannot model this: it waits for the app between keys.
    """
    for char in keys:
        host._driver.send_message(events.Key(_character_to_key(char), char))


def _library_access_modals(host) -> list[ConsoleLibraryAccessModal]:
    return [
        screen
        for screen in host.screen_stack
        if isinstance(screen, ConsoleLibraryAccessModal)
    ]


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_a_burst_of_library_chip_activations_opens_one_access_dialog() -> None:
    """TASK-34720: one dialog however many activations arrive at once.

    Each Space queued on the focused chip used to push its own Library
    access dialog, so a burst stacked identical dialogs: Save and Cancel on
    the top one revealed the next, which read as "the dialog stays open".
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = await _focused_library_chip(host, pilot)

        _send_key_burst(host, "    ")
        for _ in range(4):
            await pilot.pause()

        modals = _library_access_modals(host)
        assert len(modals) == 1, f"stacked {len(modals)} Library access dialogs"
        assert host.screen_stack[-1] is modals[0]

        await pilot.click("#library-access-cancel")
        await pilot.pause()
        assert host.screen_stack[-1] is console
        assert _library_access_modals(host) == []

        # The guard refuses only while a dialog is open: once it has closed,
        # the chip (focus returned to it) opens the dialog again.
        await pilot.pause()
        assert host.focused is console.query_one("#console-library-chip")
        await pilot.press("space")
        await pilot.pause()
        assert len(_library_access_modals(host)) == 1
        assert isinstance(host.screen_stack[-1], ConsoleLibraryAccessModal)


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_a_sentence_typed_on_the_library_chip_does_not_crash_the_console() -> (
    None
):
    """TASK-34720: the live crash's input, end to end on the real ChatScreen.

    Live on dev, a sentence typed after closing the dialog (focus back on the
    chip) pushed one dialog per space and per Enter -- 21 here. Textual paints
    every translucent modal over the one beneath it, one nested render per
    stacked screen (about 20 Python frames each, measured), so the live app
    passed Python's recursion limit and exited with RecursionError in
    ``Compositor.render_strips``. The headless harness renders from a
    shallower stack and survives 21, so the stack itself is what this pins:
    the depth that crashed is the depth that can no longer be built.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        await _focused_library_chip(host, pilot)

        _send_key_burst(
            host,
            "What do my notes say about the project plan and the next steps "
            "for it? Reply in one short sentence please.",
        )
        host._driver.send_message(events.Key("enter", "\r"))
        for _ in range(6):
            await pilot.pause()

        # A crashed app has already exited: its error is the failure to report.
        assert host._exception is None, repr(host._exception)
        assert host.is_running
        modals = _library_access_modals(host)
        assert len(modals) == 1, f"stacked {len(modals)} Library access dialogs"
        assert host.screen_stack[-1] is modals[0]
