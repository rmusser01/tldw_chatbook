"""Ctrl+S's refusals and exclusions, boot-free (TASK-34000.40 N-33, review M-5).

ADR-031 rule (c): where an editor's Save is held, Ctrl+S says why AT the
editor and moves no focus. The Pilot files (``test_library_ctrl_s_saves_in_
every_editor*.py``) drive the common paths on the real app; these pin the
branches a real boot reaches only slowly or by racing -- the prompt and skill
conflict banners, the skill's armed-delete and busy refusals, the Folder files
error and folder-change answers, the Media forms' busy answers, and the Prompt
block editor's own Ctrl+S, which the Library target must not claim -- and,
from review R1, the Notes conflict reveal (N-4), the refusal line's release
(N-2), the explicit-only clean answer (N-3) and the analysis guard's reset
(N-6). Each one binds the REAL method to a fake that holds only the state it
reads.
"""

from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace

# Stubs first: they register the optional modules the app imports below probe.
import Tests.UI._optional_module_stubs  # noqa: F401
import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from tldw_chatbook.Library.library_notes_session import (
    NoteSaveOutcome,
    NoteSaveOutcomeKind,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SKILLS
from tldw_chatbook.UI.Library_Modules.library_media_controller import (
    LibraryMediaController,
)
from tldw_chatbook.UI.Library_Modules.library_media_state import LibraryMediaState
from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
    LibraryNotesController,
)
from tldw_chatbook.UI.Library_Modules.library_prompts_controller import (
    LibraryPromptsController,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)

BUSY = "Save unavailable — wait for the current update to finish."
NO_CHANGES = "Saved — no changes."


class _Banner:
    """A conflict banner that records being brought into view."""

    def __init__(self) -> None:
        self.reveals: list[dict] = []

    def scroll_visible(self, **kwargs) -> None:
        self.reveals.append(kwargs)


def _save_target(pane_ids: list[str], *, prompt_active: bool) -> str | None:
    focused = SimpleNamespace(id="prompt-block-content-legacy-user-1")
    focused.ancestors_with_self = [focused] + [SimpleNamespace(id=i) for i in pane_ids]
    fake = SimpleNamespace(
        focused=focused,
        _file_notes_active=lambda: False,
        _media_state=SimpleNamespace(editing=False, editing_analysis=False),
        _library_skill_editor_active=lambda: False,
        _library_prompt_editor_active=lambda: prompt_active,
        check_action=lambda _action, _params: False,
    )
    fake._library_focus_within = MethodType(LibraryScreen._library_focus_within, fake)
    return LibraryScreen._library_editor_save_target(fake)


def test_the_prompt_block_editor_keeps_its_own_ctrl_s():
    """Advanced mode embeds ``PromptBlockEditor``, whose own ``ctrl+s`` opens
    its "Save…" menu and wins the focus chain; the Library target must not
    claim focus inside it, or the footer and F1 would promise a save the key
    never makes there."""
    assert _save_target(["library-prompt-work-pane"], prompt_active=True) == "prompt"
    assert (
        _save_target(
            ["library-prompt-block-editor", "library-prompt-work-pane"],
            prompt_active=True,
        )
        is None
    )


def _prompt_fake(*, refusal: str, save_shown: bool):
    banner = _Banner()
    painted: list[tuple[str, bool]] = []
    saves: list[bool] = []
    fake = SimpleNamespace(
        query_one=lambda _selector, _type=None: SimpleNamespace(display=save_shown),
        query=lambda selector: [banner]
        if selector == "#library-prompt-conflict-copy"
        else [],
        _library_prompt_save_key_refusal=lambda: refusal,
        _update_library_prompt_status_static=lambda text, reveal=False: painted.append(
            (text, reveal)
        ),
        _save_library_prompt_from_editor=lambda reveal_receipt=False: saves.append(
            reveal_receipt
        ),
    )
    LibraryPromptsController._save_library_prompt_from_key(fake)
    return banner.reveals, painted, saves


def test_the_prompt_keys_answers_at_the_editor():
    reveals, painted, saves = _prompt_fake(refusal="conflict", save_shown=False)
    assert (reveals, painted, saves) == ([{"animate": False}], [], [])
    reveals, painted, saves = _prompt_fake(refusal=BUSY, save_shown=True)
    assert (reveals, painted, saves) == ([], [(BUSY, True)], [])
    reveals, painted, saves = _prompt_fake(refusal="", save_shown=False)
    assert (reveals, painted, saves) == ([], [(NO_CHANGES, True)], [])
    reveals, painted, saves = _prompt_fake(refusal="", save_shown=True)
    assert (reveals, painted, saves) == ([], [], [True])


def _skill_fake(**state):
    banner = _Banner()
    painted: list[str] = []
    saves: list[bool] = []
    skills_state = SimpleNamespace(
        view="editor",
        dirty=state.get("dirty", True),
        selected_skill_name="demo",
        conflict=state.get("conflict", False),
        confirming_delete=state.get("confirming_delete", False),
        mutation_in_flight=state.get("mutation_in_flight", False),
    )
    fake = SimpleNamespace(
        _skills_state=skills_state,
        _library_selected_row_id=LIBRARY_ROW_BROWSE_SKILLS,
        query=lambda selector: [banner]
        if selector == "#library-skill-conflict-copy"
        else [],
        _update_library_skill_status_static=painted.append,
        action_library_skill_save=lambda: saves.append(True),
    )
    for name in (
        "_library_skill_editor_active",
        "_library_skill_save_available",
        "_library_skill_save_key_refusal",
    ):
        setattr(fake, name, MethodType(getattr(LibraryScreen, name), fake))
    LibraryScreen._save_library_skill_from_key(fake)
    return banner.reveals, painted, saves


def test_the_skill_keys_answers_at_the_editor():
    assert _skill_fake(conflict=True) == ([{"animate": False}], [], [])
    assert _skill_fake(confirming_delete=True) == (
        [],
        ["Save unavailable — finish or cancel the delete first."],
        [],
    )
    assert _skill_fake(mutation_in_flight=True) == ([], [BUSY], [])
    assert _skill_fake(dirty=False) == ([], [NO_CHANGES], [])
    assert _skill_fake() == ([], [], [True])


def _report(state: str, *, before: str, ok: bool, transitioning: bool = False) -> list:
    said: list[str] = []
    fake = SimpleNamespace(
        _save_state=state,
        _root_transitioning=transitioning,
        _path_transitioning=False,
        _set_action_status=said.append,
    )
    LibraryFileNotesWorkspace.report_key_save(fake, before, ok)
    return said


def test_the_folder_files_key_reports_every_outcome():
    assert _report("saved", before="dirty", ok=True) == ["Saved."]
    assert _report("saved", before="saved", ok=True) == [NO_CHANGES]
    assert _report("conflict", before="dirty", ok=False) == [
        "Save refused — resolve the conflict below."
    ]
    assert _report("error", before="dirty", ok=False) == [
        "Save refused — the last save failed; your draft is kept."
    ]
    # Review M-4: a folder change under way used to refuse in silence.
    assert _report("dirty", before="dirty", ok=False, transitioning=True) == [
        "Save refused — the folder is changing; try again in a moment."
    ]


def test_the_media_forms_refuse_out_loud_and_drop_the_chip():
    """Review M-4 / I-3: the metadata form during another Media change, and
    the analysis form during its own save, say so on the form and are not
    chipped."""
    shown: list[tuple[str, str]] = []
    viewer = SimpleNamespace(show_form_status=lambda form, text: shown.append((form, text)))
    fake = SimpleNamespace(
        _media_state=SimpleNamespace(bulk_delete_in_flight=True, saving_analysis=True),
        _mounted_library_media_viewer=lambda: viewer,
    )
    LibraryScreen._save_library_media_edit_from_form(fake)
    assert shown == [("edit", "Media change in progress.")]
    refusal = LibraryScreen._library_editor_save_refusal
    assert refusal(fake, "media_edit") == "Media change in progress."
    assert refusal(fake, "media_analysis") == "Already saving."
    fake._media_state.bulk_delete_in_flight = False
    fake._media_state.saving_analysis = False
    assert refusal(fake, "media_edit") == ""
    assert refusal(fake, "media_analysis") == ""


class _Recorder(list):
    """Calls, in order; a callable stand-in for one screen seam."""

    def __call__(self, *args, **kwargs):
        self.append((args, kwargs))


def test_a_notes_conflict_refusal_reveals_the_callout_in_place():
    """Review R1 N-4 (M-3): Ctrl+S in a conflicted note says why on the
    status line, scrolls the conflict callout into view, moves no focus and
    raises no toast; the Delete confirm's refusal reveals nothing."""
    for in_conflict, refusal in (
        (True, "Save locked — resolve the note conflict below."),
        (False, "Save locked — finish or cancel the destructive action."),
    ):
        banner = _Banner()
        focused = object()
        toasts, presented = _Recorder(), _Recorder()
        fake = SimpleNamespace(
            focused=focused,
            _library_note_save_refusal=lambda refusal=refusal: refusal,
            _notes_state=SimpleNamespace(shortcut_status=""),
            _apply_library_note_presentation_state=presented,
            _library_note_session=SimpleNamespace(
                snapshot=SimpleNamespace(in_conflict=in_conflict)
            ),
            query=lambda selector, banner=banner: [banner]
            if selector == "#library-note-conflict-copy"
            else [],
            notify=toasts,
            _show_library_note_shortcut_refusal=toasts,
        )
        asyncio.run(LibraryScreen.action_library_notes_save(fake))
        assert fake._notes_state.shortcut_status == refusal
        assert len(presented) == 1
        assert banner.reveals == ([{"animate": False}] if in_conflict else [])
        assert fake.focused is focused
        assert toasts == [], toasts


def test_the_media_change_refusal_line_clears_when_the_interlock_releases():
    """Review R1 N-2: "Media change in progress." under the metadata form
    goes when the interlock it named is released -- and the form's Ctrl+S
    is no longer refused."""
    shown: list[tuple[str, str]] = []
    viewer = SimpleNamespace(show_form_status=lambda form, text: shown.append((form, text)))
    state = SimpleNamespace(
        mutation_scope=None,
        mutation_authority=None,
        lifecycle_generation=0,
        bulk_delete_in_flight=True,
        saving_analysis=False,
    )
    fake = SimpleNamespace(
        _media_state=state,
        _mounted_library_media_viewer=lambda: viewer,
        _sync_library_media_browse_state=lambda *_args: None,
    )
    assert LibraryScreen._library_editor_save_refusal(fake, "media_edit")
    LibraryScreen._complete_library_media_mutation(fake)
    assert state.bulk_delete_in_flight is False
    assert shown == [("edit", "")], shown
    assert LibraryScreen._library_editor_save_refusal(fake, "media_edit") == ""


def _note_outcome_fake():
    presented: list[str] = []
    fake = SimpleNamespace(
        _library_note_session=SimpleNamespace(snapshot=SimpleNamespace(note_id="n-1")),
        _selected_note_id="n-1",
        _library_note_shortcut_status="",
        _apply_library_note_saved_presentation=lambda _outcome: None,
        is_mounted=True,
    )
    fake._apply_library_note_presentation_state = lambda: presented.append(
        fake._library_note_shortcut_status
    )
    return fake, presented


def test_only_an_explicit_note_save_answers_no_changes():
    """Review R1 N-3: an autosave can reach ACKNOWLEDGED only by racing a
    save that already landed (``mutate`` marks every change dirty, so a
    type-then-undo is a new revision and autosave writes it); that autosave
    says nothing. A Save press or Ctrl+S on a clean note answers."""
    ack = NoteSaveOutcome(
        NoteSaveOutcomeKind.ACKNOWLEDGED, note_id="n-1", message=NO_CHANGES
    )
    fake, presented = _note_outcome_fake()
    LibraryNotesController._apply_library_note_save_outcome(fake, ack)
    assert (fake._library_note_shortcut_status, presented) == ("", [])
    fake, presented = _note_outcome_fake()
    LibraryNotesController._apply_library_note_save_outcome(fake, ack, explicit=True)
    assert (fake._library_note_shortcut_status, presented) == (NO_CHANGES, [NO_CHANGES])


async def test_the_note_save_worker_tells_the_outcome_who_asked():
    """Review R1 N-3: the one save worker passes ``explicit`` through, so
    the autosave's clean outcome is presented without the clean answer."""
    ack = NoteSaveOutcome(NoteSaveOutcomeKind.ACKNOWLEDGED, note_id="n-1")
    asked: list[object] = []

    async def request_save(*, explicit):
        return ack

    fake = SimpleNamespace(
        _notes_state=SimpleNamespace(
            autosave_generation=3,
            view="editor",
            selected_note_id="n-1",
            shortcut_status="",
            autosave_state="idle",
            autosave_burst=None,
        ),
        _library_note_session=SimpleNamespace(
            snapshot=SimpleNamespace(note_id="n-1", body="b"),
            request_save=request_save,
        ),
        _update_library_note_meta_static=lambda content: None,
        _apply_library_note_save_outcome=lambda _outcome, **kw: asked.append(
            kw.get("explicit")
        ),
    )
    await LibraryScreen._save_library_note(fake, explicit=False, autosave_generation=3)
    await LibraryScreen._save_library_note(fake, explicit=True)
    assert asked == [False, True]


def test_a_reader_reset_never_strands_the_analysis_save_guard():
    """Review R1 N-6: a save worker cancelled before its first step never
    runs its ``finally``; every reset of the analysis form resets the guard
    too, so later analysis Saves are not all refused "Already saving."."""
    state = LibraryMediaState()
    state.editing_analysis = True
    state.saving_analysis = True
    fake = SimpleNamespace(_media_state=state, _close_library_media_find=lambda: None)
    LibraryScreen._reset_library_media_reader_to_no_selection(fake)
    assert state.saving_analysis is False
    assert LibraryScreen._library_editor_save_refusal(fake, "media_analysis") == ""

    discard = SimpleNamespace(
        _library_media_editing_analysis=True,
        _library_media_analysis_draft="draft",
        _library_media_saving_analysis=True,
        _sync_library_media_viewer_or_recompose=lambda: None,
    )
    LibraryMediaController.handle_library_media_analysis_cancel(
        discard, SimpleNamespace(stop=lambda: None)
    )
    assert discard._library_media_saving_analysis is False
