"""The census-found modals answer Ctrl+Q as their own close does (TASK-33622.15).

``Tests/Architecture/test_guarded_modal_quit_census.py`` lists every modal that
refuses or guards its own close. Beyond the four the task named (driven on the
real app in ``test_app_quit_in_flight_modals.py``), it found seven more that
refuse to close while an operation runs, a session switch mid-commit, and the
model popover's unsaved-edit guard. None had a ``confirm_quit``, so Ctrl+Q --
a priority binding -- quit straight past each of them.

These pin each hook's decision on a stand-in for the modal: the quit walk only
calls ``confirm_quit()``, and each hook reads one flag (or the popover's edited
fields), so the stand-in carries exactly that and records what the user is
told.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Imported at collection, under the session profile: the UI conftest's autouse
# fixture otherwise imports the app lazily under each test's own profile
# (RecoveryRequired: raw_source_selection_changed), as in
# test_modal_quit_discard_guards.py.
import tldw_chatbook.app  # noqa: F401
from tldw_chatbook.Chat.console_conversation_activation import ConsoleActivationPhase
import tldw_chatbook.Widgets.confirmation_dialog as confirmation_dialog
from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
    PromptCollectionManagerModal,
)
from tldw_chatbook.UI.Watchlists_Modules.bulk_sources_modal import BulkSourcesModal
from tldw_chatbook.Widgets.Console.console_exchange_export_dialog import (
    ConsoleExchangeExportDialog,
)
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
)
from tldw_chatbook.Widgets.Console.trace_export_dialog import TraceExportDialog
from tldw_chatbook.Widgets.Persona_Widgets.buddy_character_review import (
    BuddyCharacterReviewDialog,
)
from tldw_chatbook.Widgets.quit_while_working import (
    QUIT_AGAIN_HINT,
    QUIT_ANYWAY_RISK,
    QUIT_ANYWAY_TITLE,
    STILL_WORKING_TITLE,
    refuse_quit_while_working,
)
from tldw_chatbook.Widgets.Settings_Widgets.personal_context_review_modal import (
    PersonalContextProposalReviewModal,
    PersonalContextReviewModal,
)


def _stand_in(**state: object) -> tuple[SimpleNamespace, list[tuple[str, dict]]]:
    notices: list[tuple[str, dict]] = []
    screen = SimpleNamespace(
        notify=lambda message, **kwargs: notices.append((str(message), kwargs)),
        **state,
    )
    return screen, notices


#: (modal, the flag its close refusal reads, words the notice must name)
_IN_FLIGHT = [
    (ConsoleExchangeExportDialog, "_exporting", "export"),
    (TraceExportDialog, "_writing", "export"),
    (PromptCollectionManagerModal, "_mutation_in_flight", "collection"),
    (BuddyCharacterReviewDialog, "_publishing", "character"),
    (PersonalContextProposalReviewModal, "_busy", "personal context"),
    (PersonalContextReviewModal, "_busy", "personal context"),
    (BulkSourcesModal, "_batch_posted", "sources"),
]
_IDS = [modal.__name__ for modal, _flag, _what in _IN_FLIGHT]


@pytest.mark.asyncio
@pytest.mark.parametrize(("modal", "flag", "what"), _IN_FLIGHT, ids=_IDS)
async def test_mid_operation_ctrl_q_says_still_working_and_stays(modal, flag, what):
    screen, notices = _stand_in(**{flag: True})

    assert await modal.confirm_quit(screen) is False
    assert len(notices) == 1, notices
    message, kwargs = notices[0]
    assert what in message.lower()
    assert message.endswith(QUIT_AGAIN_HINT)
    assert kwargs == {"title": STILL_WORKING_TITLE, "severity": "warning"}


@pytest.fixture
def quit_anyway(monkeypatch):
    """Record the quit-anyway prompt instead of pushing it; answer ``.answer``."""
    record = SimpleNamespace(calls=[], answer=False)

    async def _ask(screen, message: str, **copy: str) -> bool:
        record.calls.append((screen, message, copy))
        return record.answer

    monkeypatch.setattr(confirmation_dialog, "confirm_quit_discarding_edits", _ask)
    return record


@pytest.mark.asyncio
@pytest.mark.parametrize(("modal", "flag", "what"), _IN_FLIGHT, ids=_IDS)
async def test_a_repeated_ctrl_q_mid_operation_asks_to_quit_anyway(
    modal, flag, what, quit_anyway
):
    """A flag that never clears must not make the app unquittable.

    BulkSourcesModal's owner skips a covered modal's result, so its batch
    flag can stay set for good; any operation can also hang (a fork behind a
    held SQLite lock). Escape is refused there too, so before this answer
    Ctrl+Q was the only way out, and the still-working refusal closed it.
    The first Ctrl+Q only says it is still working; a later one asks.
    """
    screen, notices = _stand_in(**{flag: True})

    assert await modal.confirm_quit(screen) is False
    assert len(notices) == 1
    assert quit_anyway.calls == [], "the first Ctrl+Q only says it is still working"

    # Wait: still nothing is lost, and no second toast stacks up.
    assert await modal.confirm_quit(screen) is False
    assert len(notices) == 1
    [(asked, message, copy)] = quit_anyway.calls
    assert asked is screen
    assert what in message.lower()
    assert message.endswith(QUIT_ANYWAY_RISK)
    assert copy == {
        "title": QUIT_ANYWAY_TITLE,
        "confirm_label": "Quit anyway",
        "cancel_label": "Wait",
    }

    # Quit anyway: the user chose it, so the quit runs.
    quit_anyway.answer = True
    assert await modal.confirm_quit(screen) is True
    assert len(quit_anyway.calls) == 2


@pytest.mark.asyncio
async def test_a_new_activity_is_announced_before_anything_is_asked(quit_anyway):
    """Each distinct operation gets its own still-working notice first."""
    screen, notices = _stand_in()

    assert (
        await refuse_quit_while_working(screen, "The draft is being discarded.")
        is False
    )
    assert await refuse_quit_while_working(screen, "Draft cleanup is running.") is False
    assert [message for message, _kwargs in notices] == [
        f"The draft is being discarded. {QUIT_AGAIN_HINT}",
        f"Draft cleanup is running. {QUIT_AGAIN_HINT}",
    ]
    assert quit_anyway.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("modal", "flag", "what"), _IN_FLIGHT, ids=_IDS)
async def test_idle_modal_lets_ctrl_q_through_silently(modal, flag, what):
    del what
    screen, notices = _stand_in(**{flag: False})

    assert await modal.confirm_quit(screen) is True
    assert notices == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "stays"),
    [
        (ConsoleActivationPhase.COMMITTING, True),
        # Before the commit Escape cancels the opening, so quitting may too.
        (ConsoleActivationPhase.OPENING_CANCELLABLE, False),
        (ConsoleActivationPhase.IDLE, False),
        (ConsoleActivationPhase.FAILURE_VISIBLE, False),
    ],
    ids=lambda value: getattr(value, "name", str(value)),
)
async def test_switcher_stays_only_while_a_chat_opening_commits(phase, stays):
    screen, notices = _stand_in(_activation_phase=phase)

    assert await ConsoleSessionSwitcherModal.confirm_quit(screen) is (not stays)
    assert len(notices) == (1 if stays else 0)
    if stays:
        assert "chat" in notices[0][0].lower()


@pytest.fixture
def asked(monkeypatch):
    """Record the discard prompt instead of pushing it; answer Keep editing."""
    calls: list[tuple[object, str]] = []

    async def _keep_editing(screen, message: str, **_copy: str) -> bool:
        calls.append((screen, message))
        return False

    monkeypatch.setattr(
        confirmation_dialog, "confirm_quit_discarding_edits", _keep_editing
    )
    return calls


@pytest.mark.asyncio
async def test_model_popover_asks_before_ctrl_q_drops_its_edits(asked):
    """Esc with edits asks first (the inline guard), so Ctrl+Q asks too."""
    screen, _notices = _stand_in(
        _pick_only=False, _edited_labels=lambda: ("Temperature", "Max tokens")
    )

    assert await ConsoleModelPopover.confirm_quit(screen) is False
    assert asked == [(screen, "2 unsaved edits to this chat: Temperature, Max tokens.")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        {"_pick_only": False, "_edited_labels": lambda: ()},
        # A pick-only popover edits nothing, whatever the draft holds.
        {"_pick_only": True, "_edited_labels": lambda: ("Temperature",)},
    ],
    ids=["nothing-edited", "pick-only"],
)
async def test_model_popover_with_nothing_to_lose_quits_without_asking(asked, state):
    screen, _notices = _stand_in(**state)

    assert await ConsoleModelPopover.confirm_quit(screen) is True
    assert asked == []
