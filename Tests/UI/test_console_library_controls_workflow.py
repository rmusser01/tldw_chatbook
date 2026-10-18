"""Joined production-UI workflows for Console Library controls."""

from __future__ import annotations

import pytest

# The harness mounts the real ChatScreen/config chain; per-test env redirect
# trips the config-participant admission (sibling strip suites' opt-in).
pytestmark = pytest.mark.bootstrap_profile
from textual.app import ComposeResult
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
from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
    build_composer_menu_entries,
)
from tldw_chatbook.Widgets.Console.console_library_access_modal import (
    ConsoleLibraryAccessModal,
    ConsoleLibraryPolicySaveOutcome,
)
from tldw_chatbook.Widgets.Console.console_library_search_modal import (
    ConsoleLibrarySearchModal,
)
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleStatusChips,
    ConsoleStatusText,
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
        await pilot.pause()
        # TASK-35160 follow-up: the Library policy readout LEFT the status
        # line (honestly: the four-state label is no longer rendered on the
        # strip). Its exact-label contract lives on the control state -- the
        # readout's renderer seam -- and the strip's former click route is
        # the composer menu's "Library access..." entry, which opens the
        # same two-axis editor the modal assertions below exercise.
        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        line = strip.query_one("#console-status-text", ConsoleStatusText)
        assert "Library" not in line.renderable.plain
        assert (
            ConsoleControlState.from_values(library_policy=snapshot).rag_label
            == label
        )
        menu_labels = [entry.label for entry in build_composer_menu_entries()]
        assert "Library access…" in menu_labels

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
