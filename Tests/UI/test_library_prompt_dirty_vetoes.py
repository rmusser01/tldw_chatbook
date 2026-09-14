"""task-32461: the three sibling dirty-Prompt vetoes must not refuse in silence.

task-32393 fixed Escape/Back. Three more seams read the SAME flag
(``_flush_library_prompt_save`` is exactly ``not _prompts_state.dirty``) and
returned without a word: the prompt-row switch, entering Select mode, and a
deep link arriving through ``_open_library_item_by_id``. The rail-row switch
(``handle_library_rail_row``) and the app-level navigation guard
(``flush_pending_work``) already notify, so these pins assert the same copy
rather than a third variant -- each fails if its seam becomes a silent no-op
again.

The deep link is the one that does not reuse the copy verbatim: per the
controller's ruling it NAMES the prompt it refused to open (and queues
nothing), because the press that caused it happened on another surface.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_library_prompt_dirty_escape import (
    WIDE_TEST_SIZE,
    _prompts_app,
)
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_MODE
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_PROMPTS
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_PROMPT_DIRTY_VETO_COPY,
    LIBRARY_PROMPT_ENTRY_DIRTY_VETO_COPY,
)

pytestmark = pytest.mark.asyncio


async def _open_prompt_list(host, pilot, prompt_id):
    """Land on the Prompts list with the seeded row painted."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-prompts").press()
    await _wait_for_selector(screen, pilot, f"#library-prompt-row-{prompt_id}")
    return screen


async def test_switching_prompt_rows_while_dirty_says_why_it_refused(
    tmp_path,
) -> None:
    """AC#1: the row press states the reason and the next step, on one line."""
    app, prompts_db, prompt_id = _prompts_app(tmp_path)
    host = LibraryHarness(app)
    notices: list[tuple[str, dict]] = []

    try:
        async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
            screen = await _open_prompt_list(host, pilot, prompt_id)
            # The exact flag the veto reads -- what the editor sets on the
            # first keystroke (see test_library_prompt_dirty_escape).
            screen._prompts_state.dirty = True
            app.notify = lambda message, **kwargs: notices.append(
                (str(message), kwargs)
            )

            screen.query_one(f"#library-prompt-row-{prompt_id}").press()
            await pilot.pause()

            assert screen._prompts_state.view != "editor", (
                "the dirty veto itself is deliberate and must still hold"
            )
            assert notices, "the row switch refused and said nothing at all"
            message, kwargs = notices[0]
            assert message == LIBRARY_PROMPT_DIRTY_VETO_COPY, message
            assert kwargs.get("severity") == "warning", kwargs
    finally:
        prompts_db.close_connection()


async def test_entering_select_mode_while_dirty_says_why_it_refused(
    tmp_path,
) -> None:
    """AC#2: pressing Select on a dirty editor explains the refusal."""
    app, prompts_db, prompt_id = _prompts_app(tmp_path)
    host = LibraryHarness(app)
    notices: list[tuple[str, dict]] = []

    try:
        async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
            screen = await _open_prompt_list(host, pilot, prompt_id)
            select = screen.query_one("#library-prompts-select")
            assert not select.disabled, "Select must be pressable to be vetoed"
            screen._prompts_state.dirty = True
            app.notify = lambda message, **kwargs: notices.append(
                (str(message), kwargs)
            )

            select.press()
            await pilot.pause()

            assert not screen._prompts_state.select_mode, (
                "the dirty veto itself is deliberate and must still hold"
            )
            assert notices, "Select refused and said nothing at all"
            message, kwargs = notices[0]
            assert message == LIBRARY_PROMPT_DIRTY_VETO_COPY, message
            assert kwargs.get("severity") == "warning", kwargs
    finally:
        prompts_db.close_connection()


async def test_a_deep_link_into_a_prompt_while_dirty_names_what_it_did_not_open(
    tmp_path,
) -> None:
    """AC#3: the entry-reconcile veto explains and names the blocked target.

    The controller's ruling: explain, do not queue. So the toast names the
    prompt that did not open and the way out, and the deep link is dropped
    exactly as before -- ``_open_library_item_by_id`` still returns ``None``
    and the editor never opens on it.
    """
    app, prompts_db, prompt_id = _prompts_app(tmp_path)
    host = LibraryHarness(app)
    notices: list[tuple[str, dict]] = []

    try:
        async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
            screen = await _open_prompt_list(host, pilot, prompt_id)
            screen._prompts_state.dirty = True
            app.notify = lambda message, **kwargs: notices.append(
                (str(message), kwargs)
            )

            result = await screen._open_library_item_by_id(
                "prompt", str(prompt_id), display_name="Meeting summarizer"
            )
            await pilot.pause()

            assert result is None, "the veto itself is deliberate and must hold"
            assert screen._prompts_state.view != "editor", (
                "nothing may be queued to open after the save"
            )
            assert notices, "the deep link evaporated and said nothing at all"
            message, kwargs = notices[0]
            # Review F-5: both hand-reachable callers hold the title, so the
            # refusal names what was pressed rather than a DB row id.
            assert message == LIBRARY_PROMPT_ENTRY_DIRTY_VETO_COPY.format(
                target="Meeting summarizer"
            ), message
            # The blocked target and the way out travel together, on one line.
            assert "Save or Discard" in message, message
            assert kwargs.get("severity") == "warning", kwargs

            # No title from the caller: the id label is the fallback.
            notices.clear()
            assert (
                await screen._open_library_item_by_id("prompt", str(prompt_id))
                is None
            )
            await pilot.pause()
            assert notices[0][0] == LIBRARY_PROMPT_ENTRY_DIRTY_VETO_COPY.format(
                target=f"Prompt {prompt_id}"
            ), notices[0][0]

            # Review F-8: a malformed deep link is dropped without naming its
            # garbage back at the user.
            notices.clear()
            assert await screen._open_library_item_by_id("prompt", "not-an-id") is None
            await pilot.pause()
            assert not notices, notices
    finally:
        prompts_db.close_connection()


async def test_navigating_into_library_while_dirty_says_why_it_refused(
    tmp_path,
) -> None:
    """AC#5: the FOURTH sibling — the one route that actually fires today.

    Task review, fix round 1: ``_flush_library_navigation_sources``
    (`library_inspection_admission.py`) is the barrier every navigation INTO
    Library crosses — a palette route (Notes/Search/Prompts all alias to
    Library), a Console hand-off, a legacy route alias. It refused on a dirty
    prompt and returned ``False`` five lines above a skill veto that speaks,
    and it returns before the deep-link seam is ever reached. Driven here
    through ``apply_navigation_context``, the same entry ``app.py`` uses for a
    same-screen route.
    """
    app, prompts_db, prompt_id = _prompts_app(tmp_path)
    host = LibraryHarness(app)
    notices: list[tuple[str, dict]] = []

    try:
        async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
            screen = await _open_prompt_list(host, pilot, prompt_id)
            screen.query_one(f"#library-prompt-row-{prompt_id}").press()
            await _wait_for_selector(screen, pilot, "#library-prompt-name")
            screen._prompts_state.dirty = True
            app.notify = lambda message, **kwargs: notices.append(
                (str(message), kwargs)
            )

            screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_MODE: "notes"})
            await _wait_for_condition(
                pilot,
                lambda: bool(notices),
                message="the route into Library refused and said nothing at all",
            )

            assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_PROMPTS, (
                "the dirty veto itself is deliberate and must still hold"
            )
            message, kwargs = notices[0]
            assert message == LIBRARY_PROMPT_DIRTY_VETO_COPY, message
            assert kwargs.get("severity") == "warning", kwargs
    finally:
        prompts_db.close_connection()
