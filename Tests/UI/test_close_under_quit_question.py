"""A dialog that finishes under "Quit while still working?" closes after Wait.

TASK-33622.15 review. From the second Ctrl+Q on, a dialog that is still
working is covered by "Quit while still working?". Several such dialogs close
themselves once their operation finishes -- the fork dialog, the Trace
export, the personal-context reviews (a save, a proposal, a cleanup retry),
the Console Prompts apply, the session switcher's fallback and
Library-recovery opens. Only the top screen may be dismissed (ADR-031), so a
close that landed while the question was up was refused and never tried
again: after Wait the dialog stayed open over finished work, some with Escape
still refused, others closing with no result so the export or review was
never reported.

``SafeModalDismissMixin.dismiss_safe_once_when_on_top`` keeps a close that
was refused only because the dialog is covered and finishes it once the
dialog is on top again -- or, covered again before that replay runs, on the
next resume. Each dialog test here raises the real question through the
dialog's own ``confirm_quit``, as the quit walk does, finishes the operation
under it, and answers Wait. The fork dialog's and the profile interview's
journeys on the real app live in ``test_app_quit_in_flight_modals.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
import threading
from pathlib import Path
from types import SimpleNamespace

from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static
from textual.worker import Worker

# Imported at collection, under the session profile: the UI conftest's autouse
# fixture otherwise imports the app lazily under each test's own profile
# (RecoveryRequired: raw_source_selection_changed), as in
# test_modal_quit_in_flight_hooks.py.
import tldw_chatbook.app  # noqa: F401
from Tests.UI.test_console_character_switcher import (
    _CharacterSwitcherApp,
    _character_row,
    _unavailable_row,
)
from Tests.UI.test_console_prompts_modal import (
    _Harness as _PromptsHarness,
    _ImprovementDriver,
    _PromptBackend,
)
from Tests.UI.test_personal_context_proposal_review import (
    _BlockingProposalService,
    _Host as _ProposalHost,
    _proposal,
)
from Tests.UI.test_personal_context_review_modal import (
    _BlockingCoordinator,
    _Coordinator as _ReviewCoordinator,
    _Host as _ReviewHost,
    _push as _push_review,
)
from Tests.UI.test_trace_responsive import _TraceHost
from Tests.UI.test_trajectory_screen import base_snapshot
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationPage,
    UnavailableCharacterReason,
)
from tldw_chatbook.Chat.console_conversation_activation import (
    ConsoleActivationPhase,
    ConsoleActivationResultKind,
    ConsoleConversationActivationResult,
)
from tldw_chatbook.Chat.console_switcher_state import SwitcherMode
from tldw_chatbook.Personal_Context.interview_coordinator import (
    InterviewCommitOutcomeUnknownError,
    InterviewCommitReceipt,
)
from tldw_chatbook.Prompt_Management.prompt_improvement_models import (
    PromptImprovementOutcome,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Console import trace_export_dialog as export_dialog_module
from tldw_chatbook.Widgets.Console.trace_export_dialog import TraceExportDialog
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
from tldw_chatbook.Widgets.quit_while_working import (
    QUIT_ANYWAY_TITLE,
    refuse_quit_while_working,
)
from tldw_chatbook.Widgets.Settings_Widgets.personal_context_review_modal import (
    ProposalReviewResult,
    ReviewCommitResult,
    ReviewCommitUnknownResult,
)


async def _until(pilot, predicate, what: str, timeout: float = 5.0) -> None:
    """Pump the app until ``predicate()`` holds, or fail naming ``what``."""
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError as exc:
        raise AssertionError(f"timed out waiting for {what}") from exc


async def _until_closed(
    pilot,
    app: App,
    modal: ModalScreen,
    delivered: Callable[[], bool],
    what: str = "the kept close",
) -> None:
    """Wait until ``modal`` left the stack and its opener has the result.

    Textual pops the screen at once but hands the result to the opener's
    callback on a later turn (``call_next``), so checking the stack alone
    can read the result before it lands.
    """
    await _until(pilot, lambda: modal not in app.screen_stack and delivered(), what)


def _shown(screen: ModalScreen, selector: str) -> bool:
    """Whether ``selector`` on ``screen`` is laid out, so a click can land.

    A screen is on the stack before its widgets are placed; Pilot aims a
    click at the widget's region, which is empty until then, so a click made
    too early lands on the backdrop. Seen once under load: Quit anyway was
    clicked before the question's buttons were placed and never answered.
    """
    found = screen.query(selector)
    return bool(found) and found.first().region.area > 0


async def _ask_quit_anyway(
    app: App, pilot, modal: ModalScreen
) -> tuple[ConfirmationDialog, Worker[bool]]:
    """Press Ctrl+Q twice over a working dialog, as the quit walk asks it.

    Args:
        app: The running app.
        pilot: Its pilot.
        modal: The dialog whose operation is still running.

    Returns:
        The "Quit while still working?" question on top of ``modal``, and the
        worker awaiting its answer.
    """
    assert await modal.confirm_quit() is False, "the first Ctrl+Q only says so"
    worker = app.run_worker(modal.confirm_quit(), exit_on_error=False)
    await _until(
        pilot,
        lambda: (
            isinstance(app.screen, ConfirmationDialog)
            and app.screen.title == QUIT_ANYWAY_TITLE
            and _shown(app.screen, "#cancel-button")
            and _shown(app.screen, "#confirm-button")
        ),
        "the quit-anyway question over the working dialog",
    )
    question = app.screen
    assert isinstance(question, ConfirmationDialog)
    assert modal in app.screen_stack
    return question, worker


async def _answer_wait(pilot, worker: Worker[bool]) -> None:
    """Choose Wait and check the quit walk was told to stay."""
    await pilot.click("#cancel-button")
    await _until(pilot, lambda: worker.is_finished, "Wait to answer the question")
    assert worker.error is None
    assert worker.result is False


# --- The shared primitive ------------------------------------------------------


class _WorkingModal(SafeModalDismissMixin, ModalScreen[str]):
    """A dialog that refuses Ctrl+Q while ``working``, like the fork dialog."""

    def __init__(self) -> None:
        super().__init__()
        self.working = True

    def compose(self) -> ComposeResult:
        yield Static("working")

    async def confirm_quit(self) -> bool:
        if not self.working:
            return True
        return await refuse_quit_while_working(self, "The thing is still running.")


class _Host(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[object] = []

    def compose(self) -> ComposeResult:
        yield Static("base")


async def test_a_close_refused_under_the_question_finishes_after_wait():
    app = _Host()
    async with app.run_test(size=(100, 30)) as pilot:
        base = app.screen
        modal = _WorkingModal()
        await app.push_screen(modal, app.results.append)
        await _until(pilot, lambda: app.screen is modal, "the working dialog")
        question, worker = await _ask_quit_anyway(app, pilot, modal)

        modal.working = False
        assert modal.dismiss_safe_once_when_on_top("finished") is False
        await pilot.pause(0.2)
        assert app.screen is question, "the close must not pop the question"
        assert app.results == []

        await _answer_wait(pilot, worker)
        await _until_closed(pilot, app, modal, lambda: bool(app.results))
        assert app.results == ["finished"]
        assert app.screen is base
        assert app._exception is None


async def test_quit_anyway_still_quits_when_a_close_was_kept():
    app = _Host()
    async with app.run_test(size=(100, 30)) as pilot:
        modal = _WorkingModal()
        await app.push_screen(modal, app.results.append)
        await _until(pilot, lambda: app.screen is modal, "the working dialog")
        _question, worker = await _ask_quit_anyway(app, pilot, modal)

        modal.working = False
        assert modal.dismiss_safe_once_when_on_top("finished") is False
        await pilot.click("#confirm-button")  # Quit anyway
        await _until(pilot, lambda: worker.is_finished, "Quit anyway to answer")
        assert worker.result is True
        await _until_closed(pilot, app, modal, lambda: bool(app.results))
        assert app.results == ["finished"]


class _Cover(ModalScreen[None]):
    """A plain screen pushed over the dialog, with no hooks of its own."""

    def compose(self) -> ComposeResult:
        yield Static("cover")


async def test_a_kept_close_covered_again_before_it_runs_waits_for_the_next_resume():
    """The replay is queued on ScreenResume but runs after something covered
    the dialog again -- as when Quit anyway's walk pushes the next prompt.
    Dropping the close there would leave the dialog open for good."""
    app = _Host()
    async with app.run_test(size=(100, 30)) as pilot:
        base = app.screen
        modal = _WorkingModal()
        await app.push_screen(modal, app.results.append)
        await _until(pilot, lambda: app.screen is modal, "the working dialog")
        first = _Cover()
        await app.push_screen(first)
        await _until(pilot, lambda: app.screen is first, "the first cover")
        assert modal.dismiss_safe_once_when_on_top("finished") is False

        # Record each replay and whether the dialog was on top when it ran.
        replays: list[bool] = []
        replay = modal._finish_pending_close

        def recording_replay() -> None:
            replays.append(app.screen is modal)
            replay()

        modal._finish_pending_close = recording_replay  # type: ignore[method-assign]

        # Uncover (ScreenResume queues the replay) and cover again before it runs.
        second = _Cover()
        app.pop_screen()
        app.push_screen(second)
        await _until(pilot, lambda: bool(replays), "the replay the first resume queued")
        await pilot.pause(0.2)
        assert replays == [False], "the replay must have run while covered again"
        assert modal in app.screen_stack and app.results == []

        second.dismiss(None)
        await _until_closed(
            pilot, app, modal, lambda: bool(app.results), "the close kept for later"
        )
        assert replays == [False, True]
        assert app.results == ["finished"]
        assert app.screen is base
        assert app._exception is None


async def test_an_uncovered_close_is_immediate_and_happens_once():
    app = _Host()
    async with app.run_test(size=(100, 30)) as pilot:
        base = app.screen
        beneath = _WorkingModal()
        await app.push_screen(beneath)
        modal = _WorkingModal()
        await app.push_screen(modal, app.results.append)
        await _until(pilot, lambda: app.screen is modal, "the dialog on top")

        assert modal.dismiss_safe_once_when_on_top("now") is True
        await _until(pilot, lambda: modal not in app.screen_stack, "its close")
        assert modal.dismiss_safe_once_when_on_top("stale") is False
        await pilot.pause(0.2)

        assert app.results == ["now"]
        assert app.screen is beneath, "a stale close must not pop the screen beneath"
        assert app.screen_stack[-2] is base


# --- The dialogs that close themselves once their work finishes ----------------


async def test_a_trace_export_written_under_the_question_reports_it_after_wait(
    tmp_path: Path, monkeypatch
):
    """Covered, the export's close was refused; Escape then closed with None,
    so the Trace screen never said the file was written."""
    target = tmp_path / "shared-trace.json"
    release = threading.Event()

    def write_when_released(destination: Path, _payload: object) -> Path:
        assert release.wait(5)
        return destination

    monkeypatch.setattr(
        export_dialog_module, "write_trajectory_export", write_when_released
    )
    app = _TraceHost()
    results: list[Path | None] = []
    async with app.run_test(size=(80, 24)) as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog, results.append)
        await pilot.pause()
        dialog.query_one("#trace-export-path", Input).value = str(target)
        await pilot.click("#trace-export-submit")
        await _until(pilot, lambda: dialog._writing, "the export to start writing")
        _question, worker = await _ask_quit_anyway(app, pilot, dialog)

        release.set()
        await _until(pilot, lambda: not dialog._writing, "the export to finish")
        await pilot.pause(0.2)
        assert results == [], "covered, the dialog cannot close yet"

        await _answer_wait(pilot, worker)
        await _until_closed(pilot, app, dialog, lambda: bool(results))
        assert results == [target]


async def test_a_proposal_resolved_under_the_question_returns_its_result_after_wait():
    """Covered, the review's close was refused and its buttons came back; the
    proposal was already accepted, and Escape closed with no result."""
    proposal = _proposal()
    service = _BlockingProposalService(proposal)
    app = _ProposalHost()
    async with app.run_test(size=(100, 32)) as pilot:
        from tldw_chatbook.Widgets.Settings_Widgets.personal_context_review_modal import (
            PersonalContextProposalReviewModal,
        )

        modal = PersonalContextProposalReviewModal(
            service, proposal=proposal, scope_label="Global"
        )
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        await pilot.click("#personal-context-proposal-accept")
        assert service.entered.wait(1)
        _question, worker = await _ask_quit_anyway(app, pilot, modal)

        service.release.set()
        await _until(pilot, lambda: not modal._busy, "the acceptance to finish")
        await pilot.pause(0.2)
        assert app.results == [], "covered, the review cannot close yet"

        await _answer_wait(pilot, worker)
        await _until_closed(pilot, app, modal, lambda: bool(app.results))
        assert app.results == [
            ProposalReviewResult(
                proposal_id=proposal.proposal_id,
                state="accepted",
                record_id="record-proposed",
            )
        ]


async def test_a_review_committed_under_the_question_closes_after_wait():
    coordinator = _BlockingCoordinator()
    app = _ReviewHost()
    async with app.run_test(size=(110, 34)) as pilot:
        from tldw_chatbook.Widgets.Settings_Widgets.personal_context_review_modal import (
            PersonalContextReviewModal,
        )

        modal = PersonalContextReviewModal(
            coordinator, session_id="session-1", diff=coordinator.diff
        )
        await _push_review(app, modal)
        await pilot.click("#personal-context-review-save-use")
        assert coordinator.commit_entered.wait(1)
        _question, worker = await _ask_quit_anyway(app, pilot, modal)

        coordinator.release_commit.set()
        await _until(pilot, lambda: not modal._busy, "the commit to finish")
        await pilot.pause(0.2)
        assert app.results == [], "covered, the review cannot close yet"

        await _answer_wait(pilot, worker)
        await _until_closed(pilot, app, modal, lambda: bool(app.results))
        assert app.results == [
            ReviewCommitResult(coordinator.receipt, enable_runtime=True)
        ]


class _HeldCleanupCoordinator(_ReviewCoordinator):
    """Saves at once with draft cleanup left pending; holds the cleanup retry.

    With ``outcome_unknown`` the save instead ends with its outcome unknown,
    which also leaves only the cleanup retry.
    """

    def __init__(self, *, outcome_unknown: bool = False) -> None:
        super().__init__()
        self.outcome_unknown = outcome_unknown
        self.receipt = InterviewCommitReceipt(
            committed_record_ids=("record-1",),
            runtime_update_succeeded=True,
            draft_cleanup_succeeded=False,
            draft_cleanup_retry_required=True,
        )
        self.cleanup_entered = threading.Event()
        self.release_cleanup = threading.Event()

    def commit(self, session_id, **kwargs):
        if self.outcome_unknown:
            self.commits.append({"session_id": session_id, **kwargs})
            raise InterviewCommitOutcomeUnknownError()
        return super().commit(session_id, **kwargs)

    def retry_draft_cleanup(self, session_id):
        self.cleanup_entered.set()
        assert self.release_cleanup.wait(5)
        return super().retry_draft_cleanup(session_id)


async def _review_retrying_cleanup(
    app: _ReviewHost, pilot, coordinator: _HeldCleanupCoordinator
):
    """Save, leave only the cleanup retry, and start it, held mid-flight."""
    from tldw_chatbook.Widgets.Settings_Widgets.personal_context_review_modal import (
        PersonalContextReviewModal,
    )

    modal = PersonalContextReviewModal(
        coordinator, session_id="session-1", diff=coordinator.diff
    )
    await _push_review(app, modal)
    await pilot.click("#personal-context-review-save-use")
    await _until(
        pilot,
        lambda: (
            (modal._receipt is not None or modal._commit_unknown) and not modal._busy
        ),
        "the save to leave the cleanup retry",
    )
    assert app.results == [], "cleanup pending, the review stays open"
    retry = modal.query_one("#personal-context-review-retry-cleanup", Button)
    assert not retry.disabled
    retry.press()
    await _until(pilot, coordinator.cleanup_entered.is_set, "the cleanup retry")
    return modal


async def test_a_cleanup_finished_under_the_question_closes_the_review_after_wait():
    """``_cleanup_finished`` after a save whose cleanup was left pending."""
    coordinator = _HeldCleanupCoordinator()
    app = _ReviewHost()
    async with app.run_test(size=(110, 34)) as pilot:
        modal = await _review_retrying_cleanup(app, pilot, coordinator)
        _question, worker = await _ask_quit_anyway(app, pilot, modal)

        coordinator.release_cleanup.set()
        await _until(pilot, lambda: not modal._busy, "the cleanup to finish")
        await pilot.pause(0.2)
        assert app.results == [], "covered, the review cannot close yet"

        await _answer_wait(pilot, worker)
        await _until_closed(pilot, app, modal, lambda: bool(app.results))
        assert app.results == [
            ReviewCommitResult(
                replace(
                    coordinator.receipt,
                    draft_cleanup_succeeded=True,
                    draft_cleanup_retry_required=False,
                ),
                enable_runtime=True,
            )
        ]


async def test_a_cleanup_after_an_unknown_save_closes_the_review_after_wait():
    """``_cleanup_finished`` after a save whose outcome is unknown."""
    coordinator = _HeldCleanupCoordinator(outcome_unknown=True)
    app = _ReviewHost()
    async with app.run_test(size=(110, 34)) as pilot:
        modal = await _review_retrying_cleanup(app, pilot, coordinator)
        assert modal._commit_unknown
        _question, worker = await _ask_quit_anyway(app, pilot, modal)

        coordinator.release_cleanup.set()
        await _until(pilot, lambda: not modal._busy, "the cleanup to finish")
        await pilot.pause(0.2)
        assert app.results == [], "covered, the review cannot close yet"

        await _answer_wait(pilot, worker)
        await _until_closed(pilot, app, modal, lambda: bool(app.results))
        assert app.results == [ReviewCommitUnknownResult()]


async def test_a_prompts_apply_finished_under_the_question_closes_after_wait():
    """Covered, the applied prompt's close was refused: after Wait the Prompts
    modal stayed open on "Applied to the Console." over a finished apply."""
    driver = _ImprovementDriver(
        PromptImprovementOutcome(
            request_id="ignored", kind="success", rewritten_prompt="Applied"
        )
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def apply(_result, _snapshot):
        started.set()
        await release.wait()
        return SimpleNamespace(kind="applied", user_message="")

    kwargs = driver.kwargs()
    kwargs["apply_improvement_result"] = apply
    app = _PromptsHarness(_PromptBackend(), improvement_kwargs=kwargs)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.enter_mode("improve")
        modal.query_one("#console-prompts-auto-improve", Button).press()
        await _until(pilot, started.is_set, "the apply to start")
        assert modal._apply_in_progress
        _question, worker = await _ask_quit_anyway(app, pilot, modal)

        release.set()
        await _until(pilot, lambda: not modal._apply_in_progress, "the apply to finish")
        await pilot.pause(0.2)
        assert app.results == [], "covered, the Prompts modal cannot close yet"

        await _answer_wait(pilot, worker)
        await _until_closed(pilot, app, modal, lambda: bool(app.results))
        assert len(app.results) == 1
        assert app.results[0].kind == "apply"
        assert app.results[0].user_text == "Applied"


async def test_a_switcher_open_committed_under_the_question_closes_after_wait():
    """The switcher's fallback close for an activator that opened the chat
    without the synchronous reveal; it stayed COMMITTING, Escape refused."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def character_loader(**_kwargs):
        return CharacterConversationPage(
            (_character_row("conversation-a", "Ada's plan", "2026-09-02T12:00:00Z"),),
            1,
            None,
            9,
        )

    async def activate(request, _cancellation):
        entered.set()
        await release.wait()
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.OPENED, request.target, True
        )

    async def wait_for_commit(_request):
        await entered.wait()

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        character_activate=activate,
        character_commit_waiter=wait_for_commit,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one(".console-switcher-result", Button).focus()
        await pilot.press("enter", "enter")
        await _until(
            pilot,
            lambda: modal._activation_phase is ConsoleActivationPhase.COMMITTING,
            "the open to start committing",
        )
        _question, worker = await _ask_quit_anyway(app, pilot, modal)

        release.set()
        await pilot.pause(0.3)
        assert app.result == "unset", "covered, the switcher cannot close yet"

        await _answer_wait(pilot, worker)
        await _until_closed(pilot, app, modal, lambda: app.result != "unset")
        assert app.result is None


async def test_a_switcher_library_recovery_accepted_under_the_question_closes():
    """The Library-recovery open commits, then is accepted while covered."""
    release = asyncio.Event()

    async def character_loader(**_kwargs):
        return CharacterConversationPage(
            (
                _unavailable_row(
                    "lost",
                    "Exact lost chat",
                    UnavailableCharacterReason.DELETED_CARD,
                    "2026-09-03T12:00:00Z",
                ),
            ),
            1,
            None,
            7,
        )

    async def recover(_result, *, is_current, on_commit_started):
        assert is_current() and on_commit_started()
        await release.wait()
        return True

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        character_open_library=recover,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        modal = app.screen
        await pilot.press("enter")
        await pilot.pause()
        modal.query_one("#console-switcher-recovery", Button).press()
        await _until(
            pilot,
            lambda: modal._activation_phase is ConsoleActivationPhase.COMMITTING,
            "the Library open to start committing",
        )
        _question, worker = await _ask_quit_anyway(app, pilot, modal)

        release.set()
        await pilot.pause(0.3)
        assert app.result == "unset", "covered, the switcher cannot close yet"

        await _answer_wait(pilot, worker)
        await _until_closed(pilot, app, modal, lambda: app.result != "unset")
        assert app.result is None
