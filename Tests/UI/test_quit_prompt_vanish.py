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

Review follow-up: Textual leaves the caller behind as a zombie -- still on the
stack, its result already delivered -- whose NEXT ``dismiss()`` raises
``InvalidStateError`` and takes the app down. So a ``SafeModalDismissMixin``
modal now refuses a dismiss while another screen covers it (the prompt is
never popped), and for any other modal the helper finishes the caller's
interrupted close once its prompt has vanished.
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
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

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
    """A prompt that is answered returns that answer, with no cancel notice.

    Args:
        press: The selector of the prompt button the user clicks.
        expected: The answer that button must resolve to.
    """
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
    """The hazard itself, on Textual's real pop: never a hang, always Stay.

    And never a zombie: Textual pops the prompt instead of the caller and
    fires the caller's callback, leaving the caller on the stack with a spent
    result. Its next dismiss would raise ``InvalidStateError`` and exit the
    app, so the helper finishes the close the caller asked for.
    """
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        base = app.screen
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
        # The caller's own result was delivered exactly once, and its
        # interrupted close is finished: nothing is left to re-dismiss.
        assert covered_results == ["covered closed itself"]
        await _until(
            pilot,
            lambda: covered not in app.screen_stack,
            "the covered modal's interrupted close to be finished",
        )
        assert app.screen is base
        await _until_cancelled_notice(pilot)
        assert (await _settled_notices(pilot)).count("Quit cancelled.") == 1

        await pilot.press("escape")
        await pilot.pause()
        assert app._exception is None
        assert app.is_running


async def test_a_prompt_popped_from_outside_leaves_the_covered_modal_open():
    """Negative control: only a screen whose own close went astray is closed.

    Here nothing dismissed the covered modal -- something else popped the
    prompt -- so its result is still pending and it must stay open.
    """
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        (
            covered,
            covered_results,
            prompt,
            worker,
        ) = await _open_prompt_over_a_covered_modal(app, pilot, _await_quit_prompt())

        app.pop_screen()

        assert await _settled(worker, "a prompt popped from outside") is _NO_ANSWER
        assert prompt not in app.screen_stack
        assert app.screen is covered
        assert covered_results == []

        covered.dismiss("closed for real")
        await _until(pilot, lambda: covered not in app.screen_stack, "its close")
        # The result callback is delivered with call_next, a moment later.
        await _until(
            pilot, lambda: covered_results == ["closed for real"], "its result"
        )
        assert app._exception is None


class _SafeCoveredModal(SafeModalDismissMixin, ModalScreen[str]):
    """A covered modal on the shared safe-dismissal primitive, like the switcher."""

    def compose(self):
        yield Static("safe covered modal")


async def _open_prompt_over_a_safe_modal(app, pilot):
    covered = _SafeCoveredModal()
    covered_results: list[object] = []
    await app.push_screen(covered, covered_results.append)
    await _until(pilot, lambda: app.screen is covered, "the covered modal")
    worker = app.run_worker(_await_quit_prompt()(app), exit_on_error=False)
    await _until(
        pilot,
        lambda: isinstance(app.screen, ConfirmationDialog),
        "the quit prompt on top of the covered modal",
    )
    return covered, covered_results, app.screen, worker


async def test_a_covered_safe_modal_cannot_pop_the_quit_prompt():
    """ADR-031: async self-closing may dismiss only the active top screen.

    ``SafeModalDismissMixin.dismiss`` refuses while another screen covers the
    modal, so the prompt stays up for the user to answer, the modal's result
    is not spent, and its own close works once it is on top again.
    """
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        covered, covered_results, prompt, worker = await _open_prompt_over_a_safe_modal(
            app, pilot
        )

        # A bare dismiss from the covered modal, as its own timer would.
        covered.dismiss("covered closed itself")
        # Longer than the helper's watch interval plus its grace.
        await pilot.pause(0.4)

        assert app.screen is prompt
        assert not worker.is_finished
        assert covered_results == []

        prompt.dismiss(False)  # Stay
        assert await _settled(worker, "the answered prompt") is False
        assert app.screen is covered
        assert "Quit cancelled." not in await _settled_notices(pilot)

        covered.dismiss("closed for real")
        await _until(pilot, lambda: covered not in app.screen_stack, "its close")
        await _until(
            pilot, lambda: covered_results == ["closed for real"], "its result"
        )
        assert app._exception is None


async def test_a_safe_modal_dismissed_twice_never_pops_the_screen_beneath():
    """A stale second dismiss (a late timer or worker) is refused too."""
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        beneath = _CoveredModal()
        await app.push_screen(beneath)
        modal = _SafeCoveredModal()
        modal_results: list[object] = []
        await app.push_screen(modal, modal_results.append)
        await _until(pilot, lambda: app.screen is modal, "the modal on top")

        modal.dismiss("first")
        await _until(pilot, lambda: modal not in app.screen_stack, "its close")
        await _until(pilot, lambda: modal_results == ["first"], "its result")
        modal.dismiss("stale")
        await pilot.pause()

        assert app.screen is beneath
        assert modal_results == ["first"]
        assert app._exception is None


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
    """An answer landing inside the grace after the pop wins over no-answer."""
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
    """A prompt gone from the stack with no answer resolves to ``no_answer``."""
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
    """The workflow quit prompt, popped by a covered modal, means Stay."""
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
    """The Console quit prompt, popped by a covered modal, means Stay."""
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
