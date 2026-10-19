"""Joined production-UI workflows for Console Library controls."""

from __future__ import annotations

import pytest
from textual import events

# The harness mounts the real ChatScreen/config chain; per-test env redirect
# trips the config-participant admission (sibling strip suites' opt-in).
pytestmark = pytest.mark.bootstrap_profile
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
from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
    ConsoleCharacterOption,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
    ACTION_LIBRARY_ACCESS,
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
from tldw_chatbook.Widgets.Console.console_system_prompt_modal import (
    ConsoleSystemPromptModal,
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


def _status_line(console) -> ConsoleStatusText:
    return console.query_one("#console-status-text", ConsoleStatusText)


def _send_key_burst(host, keys: str) -> None:
    """Deliver ``keys`` the way a terminal delivers typed text: one burst.

    Every key is queued before any is handled, so each one reaches the
    focused status line before the dialog its first activation opens is on
    screen. ``pilot.press`` cannot model this: it waits for the app between
    keys.
    """
    for char in keys:
        host._driver.send_message(events.Key(_character_to_key(char), char))


def _library_access_modals(host) -> list[ConsoleLibraryAccessModal]:
    return [
        screen
        for screen in host.screen_stack
        if isinstance(screen, ConsoleLibraryAccessModal)
    ]


def _dialogs_over(host, console) -> list:
    """Every screen stacked above the Console, bottom first."""
    return host.screen_stack[host.screen_stack.index(console) + 1 :]


async def _settle(pilot, rounds: int = 20) -> None:
    """Let queued requests, their off-thread reads and their pushes finish."""
    for _ in range(rounds):
        await pilot.pause(0.02)


@pytest.mark.asyncio
async def test_a_burst_of_library_access_requests_opens_one_dialog() -> None:
    """TASK-34720: one Library access dialog however many requests arrive.

    Every request used to push its own dialog: on the old Library chip a
    burst of Space/Enter stacked identical dialogs, and Save and Cancel on
    the top one revealed the next. The status line's Library segment has
    since moved to the composer menu's "Library access..." entry (TASK-35160
    follow-up); the opener keeps the rule for every request that reaches it
    before the first dialog is on screen. Four arrive here back to back.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-status-text")

        for _ in range(4):
            console._handle_console_composer_menu_choice(ACTION_LIBRARY_ACCESS)
        await _settle(pilot, rounds=4)

        modals = _library_access_modals(host)
        assert len(modals) == 1, f"stacked {len(modals)} Library access dialogs"
        assert host.screen_stack[-1] is modals[0]

        await pilot.click("#library-access-cancel")
        await _settle(pilot, rounds=4)
        assert host.screen_stack[-1] is console
        assert _library_access_modals(host) == []

        # The refusal lasts only while a dialog covers the Console.
        console._handle_console_composer_menu_choice(ACTION_LIBRARY_ACCESS)
        await _settle(pilot, rounds=4)
        assert len(_library_access_modals(host)) == 1
        assert isinstance(host.screen_stack[-1], ConsoleLibraryAccessModal)


@pytest.mark.asyncio
async def test_a_sentence_typed_on_the_status_line_does_not_crash_the_console() -> (
    None
):
    """TASK-34720: the live crash's input, end to end on the real ChatScreen.

    Closing a dialog opened from the status line hands focus back to the
    line, its cursor still on the segment that opened the dialog, so a user
    who goes on typing types onto that segment. Each space and the Enter
    activate it. Live, a sentence typed that way (then on the Library chip)
    pushed one dialog per space and per Enter, 21 here. Textual paints every
    translucent modal over the one beneath it, one nested render per stacked
    screen (about 20 Python frames each, measured), so the app passed
    Python's recursion limit and exited with RecursionError in
    ``Compositor.render_strips``. The System prompt segment is used because
    its editor opens from a worker, so every queued activation starts one
    before the first has pushed. Headless, the 21-dialog stack overflows
    only on some runs (it renders from a shallower Python stack), so the
    stack itself is what this pins: the depth that crashed can no longer be
    built.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-status-text")
        line = _status_line(console)
        line.focus()
        await pilot.pause()
        for _ in range(len(line.visible_segments)):
            if line._cursor_name == "system-prompt":
                break
            await pilot.press("right")
        assert host.focused is line
        assert line._cursor_name == "system-prompt"

        _send_key_burst(
            host,
            "What do my notes say about the project plan and the next steps "
            "for it? Reply in one short sentence please.",
        )
        host._driver.send_message(events.Key("enter", "\r"))
        await _settle(pilot)

        # A crashed app has already exited: its error is the failure to report.
        assert host._exception is None, repr(host._exception)
        assert host.is_running
        dialogs = _dialogs_over(host, console)
        assert [type(screen) for screen in dialogs] == [ConsoleSystemPromptModal], (
            f"stacked {len(dialogs)} dialogs: "
            f"{[type(screen).__name__ for screen in dialogs]}"
        )


#: Every status-line segment whose activation opens a dialog over the
#: Console. Each Enter/Space/click posts one request, so all of them stacked
#: under a burst the way the Library chip did (TASK-34720). Run, Sources and
#: Tools reveal the Inspector rail, Approvals moves focus and Temporary runs
#: an exclusive save worker: none of them pushes a screen, so they are not
#: listed. Provider and Model open Switch model, whose opener has refused a
#: covered Console since TASK-33008.3.
_DIALOG_SEGMENTS = ("provider", "model", "system-prompt", "assistant", "scope", "cost")


@pytest.mark.asyncio
@pytest.mark.parametrize("segment", _DIALOG_SEGMENTS)
async def test_a_burst_of_segment_activations_opens_one_dialog(
    segment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TASK-34720: one dialog per burst, from every dialog-opening segment.

    Four requests are posted from the status line back to back -- exactly
    what four queued Enter/Space keys on that segment post -- before the
    Console handles the first. Each used to push its own dialog. The Scope
    and Cost segments are hidden until there is a scope or a cost, but their
    requests reach the same handlers, so they are posted all the same.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-status-text")
        if segment == "assistant":
            # The picker opens only when a character card exists; seed one
            # at the read seam instead of writing a card into the profile.
            monkeypatch.setattr(
                console._character,
                "_console_character_picker_options",
                lambda: (ConsoleCharacterOption(character_id=7, name="Ada"),),
            )
        line = _status_line(console)
        request = ConsoleStatusText.SEGMENT_MESSAGES[segment]

        for _ in range(4):
            line.post_message(request())
        await _settle(pilot)

        dialogs = _dialogs_over(host, console)
        assert len(dialogs) == 1, (
            f"{segment} stacked {len(dialogs)} dialogs: "
            f"{[type(screen).__name__ for screen in dialogs]}"
        )

        # The refusal lasts only while a dialog covers the Console.
        await host.pop_screen()
        await _settle(pilot, rounds=5)
        assert host.screen_stack[-1] is console
        line.post_message(request())
        await _settle(pilot)
        reopened = _dialogs_over(host, console)
        assert [type(screen) for screen in reopened] == [type(dialogs[0])]
