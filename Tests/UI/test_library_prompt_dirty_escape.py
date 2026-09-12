"""task-32393: Escape on a dirty Prompts editor must say why it will not leave.

Live at 235x52 the key the footer named did nothing at all: the veto in
``_exit_library_prompt_editor_guarded`` is correct and deliberate, but it
returned ``False`` silently, so four Escape presses produced no exit, no notice
and no visible change -- which reads as the app having hung. These pin the two
halves of the contract: the refusal states its reason and its next step (AC#1),
and the footer chip names the same blocker rather than promising an exit the key
will not make (AC#2).
"""

from __future__ import annotations

import pytest
from textual.widgets import Input

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.Widgets.Library.library_prompts_canvas import (
    PROMPT_DISCARD_TOOLTIP_BUSY,
)
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_PROMPT_DIRTY_ESCAPE_CHIP,
    LIBRARY_PROMPT_DIRTY_VETO_COPY,
)
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)

pytestmark = pytest.mark.asyncio

WIDE_TEST_SIZE = (235, 52)


def _prompts_app(tmp_path):
    """A Library app whose Prompts canvas has one real, openable prompt."""
    app = _build_test_app()
    app.library_new_profile_admission = False
    prompts_db = PromptsDatabase(
        tmp_path / "dirty-escape-prompts.db", client_id="dirty-escape-test"
    )
    prompt_id, _uuid, _msg = prompts_db.add_prompt(
        name="Summarize",
        author="Alice",
        details="A summarizer",
        system_prompt="You are concise.",
        user_prompt="Summarize: {text}",
    )
    app.prompt_scope_service = PromptScopeService(
        local_service=LocalPromptService(prompts_db),
        server_service=None,
    )
    return app, prompts_db, prompt_id


async def _open_prompt_editor(host, pilot, prompt_id):
    """Open the editor on the seeded prompt, armed and clean."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-prompts").press()
    await _wait_for_selector(screen, pilot, f"#library-prompt-row-{prompt_id}")
    screen.query_one(f"#library-prompt-row-{prompt_id}").press()
    await _wait_for_selector(screen, pilot, "#library-prompt-name")
    # The editor arms one refresh after it mounts; before that its own mount-time
    # ``Changed`` events are (correctly) ignored, so an edit made now is not one.
    await _wait_for_condition(
        pilot,
        lambda: screen._prompts_state.editor_armed,
        message="the prompt editor never armed",
    )
    return screen


async def _dirty_the_open_editor(screen, pilot):
    """Type one character into the open editor, through the real widget."""
    name = screen.query_one("#library-prompt-name", Input)
    name.focus()
    await pilot.press("!")
    await _wait_for_condition(
        pilot,
        lambda: screen._prompts_state.dirty,
        message="typing into the editor left it clean",
    )


async def test_escape_on_a_dirty_prompt_editor_states_why_it_will_not_leave(
    tmp_path,
) -> None:
    """AC#1/AC#3: the veto is explained on the same line as the next step."""
    app, prompts_db, prompt_id = _prompts_app(tmp_path)
    host = LibraryHarness(app)
    notices: list[tuple[str, dict]] = []

    try:
        async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
            screen = await _open_prompt_editor(host, pilot, prompt_id)
            await _dirty_the_open_editor(screen, pilot)
            app.notify = lambda message, **kwargs: notices.append(
                (str(message), kwargs)
            )

            await pilot.press("escape")
            await pilot.pause()

            assert screen._prompts_state.view == "editor", (
                "the dirty veto itself is deliberate and must still hold"
            )
            assert notices, "Escape refused to leave and said nothing at all"
            message, kwargs = notices[0]
            assert message == LIBRARY_PROMPT_DIRTY_VETO_COPY, message
            assert kwargs.get("severity") == "warning", kwargs
            # The reason and the next step travel together, on one line.
            assert "Unsaved" in message and "Save or Discard" in message, message
    finally:
        prompts_db.close_connection()


async def test_the_prompt_editor_footer_chip_names_the_dirty_blocker(
    tmp_path,
) -> None:
    """AC#2/AC#3: the chip and the key agree in BOTH editor states.

    Asserted against ``_footer_shortcut_registration`` -- what the footer widget
    was actually handed -- not only the selector function, because the dirty flip
    deliberately does not recompose (that would remount the editor's fields), so
    a chip that is only correct on the next recompose is still a lie on screen.
    """
    app, prompts_db, prompt_id = _prompts_app(tmp_path)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
            screen = await _open_prompt_editor(host, pilot, prompt_id)

            # Clean: Escape genuinely leaves, and the chip says so.
            assert ("esc", "back to list") in screen._footer_shortcut_registration[1]

            await _dirty_the_open_editor(screen, pilot)

            # Dirty: Escape refuses, so the chip names the blocker instead.
            chips = screen._footer_shortcut_registration[1]
            assert ("esc", LIBRARY_PROMPT_DIRTY_ESCAPE_CHIP) in chips, chips
            assert ("esc", "back to list") not in chips, chips
    finally:
        prompts_db.close_connection()


async def test_escape_during_an_in_flight_write_says_so_too(tmp_path) -> None:
    """Review round 1 (F10): the last state where Escape neither left nor spoke.

    A save/delete worker holds ``_library_prompts_mutation_in_flight``; the
    guarded exit refused on it before reaching the dirty check, and refused
    silently. The Discard button already explains this state in its tooltip, so
    the refusal borrows that copy rather than inventing a second one.
    """
    app, prompts_db, prompt_id = _prompts_app(tmp_path)
    host = LibraryHarness(app)
    notices: list[str] = []

    try:
        async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
            screen = await _open_prompt_editor(host, pilot, prompt_id)
            screen._prompts_state.mutation_in_flight = True
            app.notify = lambda message, **kwargs: notices.append(str(message))

            await pilot.press("escape")
            await pilot.pause()

            assert screen._prompts_state.view == "editor", "the refusal must hold"
            assert notices == [PROMPT_DISCARD_TOOLTIP_BUSY], notices
    finally:
        prompts_db.close_connection()
