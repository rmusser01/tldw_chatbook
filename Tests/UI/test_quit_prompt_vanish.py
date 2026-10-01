"""A quit prompt that leaves the screen stack unanswered means Stay (TASK-33622.10).

Ctrl+Q is a priority binding, so the quit flow can push its prompts while a
modal is open. In Textual 8.2.8 ``Screen.dismiss()`` resolves its own result
and then calls ``app.pop_screen()``, which pops the app's TOP screen -- not
necessarily the caller -- and ``App.pop_screen`` drops the popped screen's
result callback without resolving it. A covered modal that closes itself from
a timer therefore pops a quit prompt sitting above it, and a bare
``push_screen_wait`` on that prompt never returns.

``await_quit_prompt`` is the one choke point every quit-flow prompt awaits.
These pin it against a mounted Textual app (the real pop semantics), plus the
ordering race no mounted app can schedule on demand. The real-``TldwCli``
journeys, one per prompt route, live in ``test_app_quit_under_modal.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.screen import ModalScreen
from textual.widgets import Static

import tldw_chatbook.Widgets.confirmation_dialog as confirmation_dialog
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

pytestmark = pytest.mark.asyncio

#: Distinct from every real answer (True, False, None), so a test can tell
#: "the prompt vanished" apart from "the user answered".
_NO_ANSWER = object()

_HANG_GUARD_SECONDS = 5.0


async def _until(pilot, predicate, what: str, timeout: float = 5.0) -> None:
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError as exc:
        raise AssertionError(f"timed out waiting for {what}") from exc


async def _settled(worker, what: str):
    """Wait for ``worker`` with a hard bound: a hang is the defect under test.

    Polls rather than ``await worker.wait()``: Textual's ``Worker.wait``
    turns the timeout's cancellation of the WAITER into ``WorkerCancelled``,
    which would hide the hang behind a misleading error.
    """
    try:
        async with asyncio.timeout(_HANG_GUARD_SECONDS):
            while not worker.is_finished:
                await asyncio.sleep(0.02)
    except TimeoutError as exc:
        raise AssertionError(f"{what} never returned (orphaned prompt)") from exc
    if worker.error is not None:
        raise worker.error
    return worker.result


class _CoveredModal(ModalScreen[str]):
    """A modal the quit prompt opens over, like the switcher or a form."""

    def compose(self):
        yield Static("covered modal")


def _prompt() -> ConfirmationDialog:
    return ConfirmationDialog(
        title="Quit?", message="Quit?", confirm_label="Quit", cancel_label="Stay"
    )


def _notices(app) -> list[str]:
    return [note.message for note in app._notifications]


async def _settled_notices(pilot) -> list[str]:
    """Notices once queued ``Notify`` messages are handled.

    ``App.notify`` posts a message, so a toast lands a moment after the call;
    reading ``_notifications`` straight away races it.
    """
    await pilot.pause()
    return _notices(pilot.app)


async def _until_cancelled_notice(pilot) -> None:
    await _until(
        pilot,
        lambda: "Quit cancelled." in _notices(pilot.app),
        'the "Quit cancelled." toast',
    )


async def _open_prompt_over_a_covered_modal(app, pilot, asking):
    """Open a covered modal, then run ``asking(app)`` until its prompt is on top.

    Returns:
        ``(covered, covered_results, prompt, worker)``.
    """
    covered = _CoveredModal()
    covered_results: list[object] = []
    await app.push_screen(covered, covered_results.append)
    await _until(pilot, lambda: app.screen is covered, "the covered modal")
    worker = app.run_worker(asking(app), exit_on_error=False)
    await _until(
        pilot,
        lambda: isinstance(app.screen, ConfirmationDialog),
        "the quit prompt on top of the covered modal",
    )
    return covered, covered_results, app.screen, worker


def _await_quit_prompt(prompt=None):
    """``asking`` callable: one fresh prompt through the choke point."""

    async def _asking(app):
        return await confirmation_dialog.await_quit_prompt(
            app, prompt or _prompt(), no_answer=_NO_ANSWER
        )

    return _asking


@pytest.mark.parametrize(
    ("press", "expected"),
    [("#confirm-button", True), ("#cancel-button", False)],
    ids=["quit", "stay"],
)
async def test_an_answered_prompt_resolves_to_the_answer(press, expected):
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        base = app.screen
        prompt = _prompt()
        worker = app.run_worker(_await_quit_prompt(prompt)(app), exit_on_error=False)
        # The prompt is on the stack the moment it is pushed; wait for its
        # button to be laid out, or the click lands on nothing.
        await _until(
            pilot,
            lambda: (
                app.screen is prompt
                and bool(prompt.query(press))
                and prompt.query_one(press).region.area > 0
            ),
            "the quit prompt's buttons",
        )

        await pilot.click(press)

        assert await _settled(worker, "an answered prompt") is expected
        assert app.screen is base
        assert "Quit cancelled." not in await _settled_notices(pilot)


async def test_a_prompt_popped_by_the_modal_it_covers_resolves_to_no_answer():
    """The hazard itself, on Textual's real pop: never a hang, always Stay."""
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        (
            covered,
            covered_results,
            prompt,
            worker,
        ) = await _open_prompt_over_a_covered_modal(app, pilot, _await_quit_prompt())

        # A bare dismiss from the covered modal, as its own timer would.
        covered.dismiss("covered closed itself")

        assert await _settled(worker, "a vanished prompt") is _NO_ANSWER
        await _until(pilot, lambda: prompt not in app.screen_stack, "the pop")
        # Textual popped the prompt, not the caller, and fired the caller's
        # own callback: exactly the hazard, reproduced on the real stack.
        assert app.screen is covered
        assert covered_results == ["covered closed itself"]
        await _until_cancelled_notice(pilot)
        assert (await _settled_notices(pilot)).count("Quit cancelled.") == 1


async def test_an_answer_followed_by_a_covered_pop_in_the_same_tick_keeps_the_answer():
    """Answered, then the covered modal also closes: the answer wins."""
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        covered, _results, prompt, worker = await _open_prompt_over_a_covered_modal(
            app, pilot, _await_quit_prompt()
        )

        prompt.dismiss(True)
        covered.dismiss("covered closed itself")

        assert await _settled(worker, "an answered-then-popped prompt") is True
        assert "Quit cancelled." not in await _settled_notices(pilot)


class _StackFirstApp:
    """Drives the one ordering a mounted app cannot schedule on demand.

    Textual 8.2.8 resolves a dismissed screen's result BEFORE it pops it, so
    on a real stack an answer is always in hand by the time the prompt is
    gone. The helper must not depend on that ordering: here the prompt leaves
    the stack first and its answer lands a moment later.
    """

    def __init__(self) -> None:
        self.screen_stack: list[object] = ["base"]
        self.answer: asyncio.Future | None = None
        self.notices: list[tuple[str, str]] = []

    def push_screen(self, screen, *, wait_for_dismiss: bool = False):
        assert wait_for_dismiss is True
        self.screen_stack.append(screen)
        self.answer = asyncio.get_running_loop().create_future()
        return self.answer

    def notify(self, message: str, *, severity: str = "information") -> None:
        self.notices.append((message, severity))


async def test_an_answer_that_lands_just_after_the_pop_still_wins(monkeypatch):
    monkeypatch.setattr(confirmation_dialog, "PROMPT_WATCH_INTERVAL_SECONDS", 0.01)
    app = _StackFirstApp()
    prompt = object()
    asking = asyncio.ensure_future(
        confirmation_dialog.await_quit_prompt(app, prompt, no_answer=_NO_ANSWER)
    )
    await asyncio.sleep(0.03)
    app.screen_stack.remove(prompt)
    await asyncio.sleep(0.02)  # the watcher has seen it go, inside the grace
    app.answer.set_result(False)

    assert await asyncio.wait_for(asking, _HANG_GUARD_SECONDS) is False
    assert app.notices == []


async def test_a_prompt_that_never_answers_after_leaving_resolves_to_no_answer(
    monkeypatch,
):
    monkeypatch.setattr(confirmation_dialog, "PROMPT_WATCH_INTERVAL_SECONDS", 0.01)
    app = _StackFirstApp()
    prompt = object()
    asking = asyncio.ensure_future(
        confirmation_dialog.await_quit_prompt(
            app, prompt, no_answer="cancel", vanished_notice=None
        )
    )
    await asyncio.sleep(0.03)
    app.screen_stack.remove(prompt)

    assert await asyncio.wait_for(asking, _HANG_GUARD_SECONDS) == "cancel"
    assert app.notices == []  # vanished_notice=None stays silent
    # The orphaned future is left alone: cancelling it would make a late
    # Screen.dismiss raise InvalidStateError inside the prompt's handler.
    assert not app.answer.done()


async def test_the_discard_and_quit_prompt_survives_the_modal_it_covers():
    """``confirm_quit_discarding_edits`` routes through the choke point."""
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        covered, _results, prompt, worker = await _open_prompt_over_a_covered_modal(
            app,
            pilot,
            lambda app: confirmation_dialog.confirm_quit_discarding_edits(
                app.screen, "You have unsaved changes."
            ),
        )
        assert prompt.title == "Discard changes and quit?"

        covered.dismiss(None)

        assert await _settled(worker, "the discard-and-quit prompt") is False
        await _until_cancelled_notice(pilot)


def _app_level_quit_prompts_app() -> App:
    """A mounted app carrying TldwCli's own app-level quit prompts.

    Built per test, not at import, so a missing seam fails only these tests.
    """

    class _AppLevelQuitPromptsApp(App):
        _confirm_workflow_session_quit = TldwCli._confirm_workflow_session_quit
        _await_console_quit_confirmation = TldwCli._await_console_quit_confirmation
        _await_quit_prompt = TldwCli._await_quit_prompt

        def __init__(self) -> None:
            super().__init__()
            review = SimpleNamespace(
                state="review", workflow_id="wf-1", revision_id="r-1"
            )
            self._workflow_session = SimpleNamespace(view=lambda: review)

    return _AppLevelQuitPromptsApp()


async def test_the_workflow_quit_prompt_survives_the_modal_it_covers():
    app = _app_level_quit_prompts_app()
    async with app.run_test(size=(100, 30)) as pilot:
        covered, _results, prompt, worker = await _open_prompt_over_a_covered_modal(
            app, pilot, lambda app: app._confirm_workflow_session_quit()
        )
        assert prompt.title == "Quit with a workflow in progress?"

        covered.dismiss(None)

        assert await _settled(worker, "the workflow quit prompt") is False
        assert app._workflow_quit_approved_view is None
        await _until_cancelled_notice(pilot)


async def test_the_console_quit_prompt_survives_the_modal_it_covers():
    app = _app_level_quit_prompts_app()
    async with app.run_test(size=(100, 30)) as pilot:
        (
            covered,
            _results,
            _prompt_on_top,
            worker,
        ) = await _open_prompt_over_a_covered_modal(
            app,
            pilot,
            lambda app: app._await_console_quit_confirmation(_prompt()),
        )

        covered.dismiss(None)

        assert await _settled(worker, "the Console quit prompt") is False
        await _until_cancelled_notice(pilot)
