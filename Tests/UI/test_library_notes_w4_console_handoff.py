"""Library ▸ Notes wave 4, group `console-handoff` (critique #3).

- task-32536: "Use in Console" on a fresh profile refused the note with two
  disagreeing messages (a toast about workspace linking, a status line about
  Console readiness), and the failure stayed on the status line of the next
  note opened. The blocker was the workspace gate (PROVEN headlessly:
  ``reason_code="not_in_active_workspace"``), so the hand-off now links the
  open note to the active workspace on the way -- the "Use as source"
  precedent (task-32107) -- and reports one line when it cannot.
- task-32555: the Console first-run card's actions were text lines with a
  text-only focus cue; the graduation toast spoke the product's words; Enter
  in an empty provider key field did nothing.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Static, TextArea

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _LibraryEvidenceGates,
    _active_library_screen,
    _build_test_app,
    _new_library_onboarding_app,
    _open_note_editor,
    _seed_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_evidence_round,
    _wait_for_library_shell,
    _wait_for_note_row,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_content_evidence import LibraryContentEvidence
from tldw_chatbook.Library.library_rail_state import LibraryLifecycle
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal

REGISTRY_MISSING_LINE = (
    "Can't use this note in Console — the workspace registry is unavailable. "
    "Next: restart Chatbook, then try again."
)


def _transfer_status(screen) -> str:
    return str(screen.query_one("#library-note-transfer-status", Static).renderable)


def _painted(screen) -> str:
    """Everything the compositor actually paints -- the user's surface."""
    return "\n".join(
        "".join(segment.text for segment in strip)
        for strip in screen._compositor.render_strips()
    )


def _notes_host(notes, *, registry: bool = True) -> tuple[LibraryHarness, object]:
    app = _build_test_app()
    _seed_conversations(app, [], notes=notes)
    app.open_chat_with_handoff = Mock()
    app.notify = Mock()
    if not registry:
        app.workspace_registry_service = None
    return LibraryHarness(app), app


# --- task-32536 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_use_in_console_links_the_open_note_and_stages_it_on_a_fresh_profile():
    """AC#1: a note nobody linked is linked to the active workspace and staged.

    The seeded note is deliberately NOT a member of the active workspace --
    the fresh-profile shape the critique hit -- so the gate is closed when
    the button is pressed (asserted as the control below).
    """
    host, app = _notes_host(_two_notes()[:1])
    registry = app.workspace_registry_service
    active = registry.get_active_workspace()
    assert active is not None
    real_link = registry.link_membership
    links: list[tuple[str, str, str]] = []

    def recording_link(workspace_id, **kwargs):
        links.append((workspace_id, kwargs["item_type"], kwargs["item_id"]))
        return real_link(workspace_id, **kwargs)

    registry.link_membership = recording_link

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)
        gate = screen._library_workspace_depth_state(refresh=True)
        assert gate.context_handoff_enabled is False, "control: the gate is closed"
        workspace_name = gate.workspace_name

        screen.query_one("#library-note-use-in-console").press()
        await pilot.pause()
        await pilot.pause()

        assert _transfer_status(screen) == (
            f"Use in Console complete — Linked to {workspace_name} · staged in Console."
        )
        assert "check Console readiness" not in _painted(screen)

    assert links == [(active.workspace_id, "note", "n-1")]
    app.open_chat_with_handoff.assert_called_once()
    assert app.open_chat_with_handoff.call_args.args[0].source_id == "n-1"
    assert not [
        call for call in app.notify.call_args_list if "Copy or link" in str(call.args[0])
    ]


@pytest.mark.asyncio
async def test_a_failed_hand_off_shows_exactly_one_message_naming_the_blocker():
    """AC#2: one line, naming the real blocker with its remedy; no toast, and no
    "Review the error" when there is no error on screen to review."""
    host, app = _notes_host(_two_notes()[:1], registry=False)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-use-in-console").press()
        await pilot.pause()
        await pilot.pause()

        assert _transfer_status(screen) == REGISTRY_MISSING_LINE
        painted = _painted(screen)
        assert "Review the error" not in painted
        assert "check Console readiness" not in painted

    app.notify.assert_not_called()
    app.open_chat_with_handoff.assert_not_called()


@pytest.mark.asyncio
async def test_a_previous_hand_off_failure_clears_on_note_change_and_on_save():
    """AC#3/#4: the failure never reaches the list, ends when another note is
    opened, and ends when the note it belongs to is saved."""
    host, app = _notes_host(_two_notes(), registry=False)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot, "n-1")

        screen.query_one("#library-note-use-in-console").press()
        await pilot.pause()
        await pilot.pause()
        assert _transfer_status(screen) == REGISTRY_MISSING_LINE
        # The task's title is two DISAGREEING messages. The editor header
        # folds the transfer row into its own line by design
        # (``_authority_copy``), so the sentence may echo there -- but every
        # surface that carries a hand-off message carries the SAME one.
        carriers = [
            str(static.renderable)
            for static in screen.query(Static)
            if "Can't use this note" in str(static.renderable)
        ]
        assert carriers, "the failure reached no surface at all"
        for text in carriers:
            assert REGISTRY_MISSING_LINE in text, text
            assert "check Console readiness" not in text, text
        # The list pane beside the editor is the navigator's surface, not
        # the editor's: the critique saw the editor's failure painted there.
        for status in screen.query("#library-notes-status"):
            assert "Can't use" not in str(status.renderable)
        app.notify.assert_not_called()

        # Note change: open the second note from the list.
        row = await _wait_for_note_row(screen, pilot, "n-2")
        row.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._notes_state.selected_note_id == "n-2",
            message="second note did not open",
        )
        await pilot.pause()
        assert _transfer_status(screen) == ""
        assert "Can't use this note" not in _painted(screen)

        # Fail again on this note, then save it: Saved, no failure text.
        screen.query_one("#library-note-use-in-console").press()
        await pilot.pause()
        assert _transfer_status(screen) == REGISTRY_MISSING_LINE
        screen.query_one("#library-note-body", TextArea).text = "bravo, edited"
        await pilot.pause()
        screen.query_one("#library-note-save").press()
        await _wait_for_condition(
            pilot,
            lambda: str(
                screen.query_one("#library-note-status", Static).renderable
            ).startswith("Saved"),
            message="explicit save did not settle",
        )
        await pilot.pause()
        assert _transfer_status(screen) == ""
        painted = _painted(screen)
        assert "Can't use this note" not in painted
        assert "failed" not in painted.lower()

    app.open_chat_with_handoff.assert_not_called()


# --- task-32555 ---------------------------------------------------------------


class _SetupModalHarness(ConsolidatedCSSApp):
    """The card under the sheets the real app loads (bundle + Console split)."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal(id="console-setup-modal")

    async def on_mount(self) -> None:
        from tldw_chatbook.Chat.console_onboarding_state import (
            ConsoleSetupCardState,
            ConsoleSetupStep,
        )

        self.query_one("#console-setup-modal", ConsoleSetupModal).sync_card_state(
            ConsoleSetupCardState(
                mode="card",
                steps=(
                    ConsoleSetupStep(state="active", label="Connect a provider"),
                    ConsoleSetupStep(state="pending", label="Pick a model"),
                    ConsoleSetupStep(state="pending", label="Send your first message"),
                ),
            ),
            action_label="Set up provider",
            action_tooltip="Open provider settings.",
        )


@pytest.mark.asyncio
async def test_console_setup_card_actions_render_as_buttons_with_a_focus_cue():
    """AC#1: the card's actions look like buttons and focus changes their shape."""
    app = _SetupModalHarness()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        actions = [
            button
            for button in app.query(".console-setup-modal-action").results(Button)
            if button.display
        ]
        assert [str(button.label) for button in actions][:2] == [
            "Set up provider",
            "Write a note in Library",
        ]
        for button in actions:
            assert button.compact is False, f"{button.label!r} is a text line"
            assert button.styles.border.top[0] == "round", (
                f"{button.label!r} has no button edge"
            )
        actions[1].focus()
        await pilot.pause()
        assert actions[1].styles.outline_left[0] == "heavy"
        assert actions[1].styles.outline_right[0] == "heavy"
        assert actions[0].styles.outline_left[0] != "heavy"


@pytest.mark.asyncio
async def test_graduation_toast_is_gone():
    """AC#2 ("or is dropped"): the rail growing is the evidence; no toast."""
    notifications: list = []
    gates = _LibraryEvidenceGates(
        outcomes={"notes": [LibraryContentEvidence.HAS_USER_CONTENT]}
    )
    app = _new_library_onboarding_app(gates, starter=True)
    app.notify = lambda message, **kwargs: notifications.append(str(message))
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = host.screen_stack[-1]
            await _wait_for_evidence_round(pilot, gates)
            gates.release_round(0, "notes")
            await _wait_for_condition(
                pilot,
                lambda: screen._library_lifecycle is LibraryLifecycle.GRADUATED,
                message="note evidence did not graduate",
            )
            await pilot.pause()
    finally:
        gates.release_all()

    assert notifications == []
