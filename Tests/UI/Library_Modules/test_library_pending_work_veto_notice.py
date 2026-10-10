"""TASK-34000.27: the sentence a nav-bar click gets when Library's flush
refuses to let the Notes editor go.

Pure. ``library_note_flush_veto_notice`` builds the toast from the typed
``NoteFlushOutcome``: a validation veto repeats the save's own message (which
already names the field and the fix, so a keyword veto never blames the
title -- review N-25), and every other kind reuses the Esc path's sentence
(`_library_note_editor_exit_veto_message`, owned by TASK-34000.29) with the
clicked destination in its head. The two formerly silent Library vetoes get
an honest generic sentence of the same shape.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_notes_session import (
    NoteFlushOutcome,
    NoteFlushOutcomeKind,
)
from tldw_chatbook.UI.Library_Modules.library_pending_work import (
    library_file_notes_flush_veto_notice,
    library_note_flush_veto_notice,
    library_prompt_mutation_veto_notice,
)
from tldw_chatbook.UI.Screens.library_screen import (
    _library_note_editor_exit_veto_message,
)

#: ``bootstrap_profile``: importing the Library screen module (for the Esc
#: path's sentence table) runs the app's recovery bootstrap, which the
#: per-test sandbox fails locally with ``RecoveryRequired``.
pytestmark = [pytest.mark.unit, pytest.mark.bootstrap_profile]

_TITLE_REASON = "Title begins or ends with whitespace — remove it to save."
_KEYWORD_REASON = (
    "Keywords contain a case-insensitive duplicate — remove one to save."
)


def test_validation_veto_repeats_the_saves_own_reason_after_the_destination():
    outcome = NoteFlushOutcome(NoteFlushOutcomeKind.VALIDATION_VETO, _TITLE_REASON)

    assert (
        library_note_flush_veto_notice(outcome, destination="Console")
        == f"Can't open Console yet: {_TITLE_REASON}"
    )


def test_keyword_veto_never_blames_the_title():
    outcome = NoteFlushOutcome(NoteFlushOutcomeKind.VALIDATION_VETO, _KEYWORD_REASON)

    text = library_note_flush_veto_notice(outcome, destination="Console")

    assert text == f"Can't open Console yet: {_KEYWORD_REASON}"
    assert "title" not in text.lower()


def test_no_destination_label_falls_back_to_cant_leave_yet():
    outcome = NoteFlushOutcome(NoteFlushOutcomeKind.VALIDATION_VETO, _TITLE_REASON)

    assert (
        library_note_flush_veto_notice(outcome, destination="")
        == f"Can't leave yet: {_TITLE_REASON}"
    )


def test_validation_veto_without_a_message_uses_the_esc_paths_sentence():
    outcome = NoteFlushOutcome(NoteFlushOutcomeKind.VALIDATION_VETO, "")

    text = library_note_flush_veto_notice(outcome, destination="Console")

    assert text.startswith("Can't open Console yet — ")
    assert text.endswith(
        _library_note_editor_exit_veto_message(
            NoteFlushOutcomeKind.VALIDATION_VETO
        ).removeprefix("Can't leave yet — ")
    )


@pytest.mark.parametrize(
    "kind",
    [
        NoteFlushOutcomeKind.FAILED,
        NoteFlushOutcomeKind.CONFLICTED,
        NoteFlushOutcomeKind.BLOCKED,
        NoteFlushOutcomeKind.STALE,
    ],
)
def test_other_kinds_reuse_the_esc_sentence_with_the_destination_head(kind):
    esc_sentence = _library_note_editor_exit_veto_message(kind)
    assert esc_sentence.startswith("Can't leave yet — "), esc_sentence
    outcome = NoteFlushOutcome(kind, "status-line wording, not the toast's")

    text = library_note_flush_veto_notice(outcome, destination="Home")

    assert text == "Can't open Home yet — " + esc_sentence.removeprefix(
        "Can't leave yet — "
    )
    assert "status-line wording" not in text


def test_a_permitted_outcome_has_nothing_to_say():
    assert (
        library_note_flush_veto_notice(
            NoteFlushOutcome(NoteFlushOutcomeKind.PERMITTED), destination="Console"
        )
        == ""
    )


def test_the_formerly_silent_vetoes_say_something_honest():
    files = library_file_notes_flush_veto_notice(destination="Console")
    prompt = library_prompt_mutation_veto_notice(destination="Console")

    assert files.startswith("Can't open Console yet — ")
    assert "Folder files" in files
    assert prompt.startswith("Can't open Console yet — ")
    assert "prompt" in prompt.lower() and "in progress" in prompt
    assert library_prompt_mutation_veto_notice(destination="").startswith(
        "Can't leave yet — "
    )
