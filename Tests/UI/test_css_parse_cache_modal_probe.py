# test_css_parse_cache_modal_probe.py
# Description: TASK-21115's "feature-rich session" arithmetic, pinned.
#
# The consolidation tour guard (test_widget_css_consolidation.py) measures the
# PLAIN 13-destination tour. The 2026-08-22 holistic review's finding was that
# the plain tour is not the exposed case: ~10 distinct modal opens on top of it
# crossed Textual's LRUCache(64) parse-cache cliff, because 25 new DEFAULT_CSS
# declarations had accreted since TASK-15450 (measured on the review pin:
# tour=47, +12 modal opens=60 -- over the 56 soft limit -- and +the full
# accreted set=70, past the cliff; post-conversion: 44/45/45).
#
# This test replays that arithmetic: it runs the same tour, then simulates the
# first-mount source registration of 12 user-openable modals plus the whole
# formerly-DEFAULT_CSS set, using exactly DOMNode._get_default_css's walk
# (textual/dom.py: one source per MRO base with DEFAULT_CSS in its own
# __dict__, keyed (getfile(base), f"{base.__name__}.DEFAULT_CSS")), and holds
# the result under the soft limit. Every listed class rides BUNDLED_CSS now,
# so each registration should add ~nothing; a class quietly regrowing a
# DEFAULT_CSS (or a new modal skipping the bundle) erodes the margin here
# while the static allowlist ratchet names the offender.
from __future__ import annotations

import asyncio
from inspect import getfile

import pytest

MODAL_TARGETS = [
    ("tldw_chatbook.Widgets.Console.console_reaction_picker_modal", "ConsoleReactionPickerModal"),
    ("tldw_chatbook.Widgets.Console.console_review_notes_modal", "ConsoleReviewNotesModal"),
    ("tldw_chatbook.Widgets.Console.console_side_chat_modal", "ConsoleSideChatModal"),
    ("tldw_chatbook.Widgets.Console.console_feedback_comment_modal", "ConsoleFeedbackCommentModal"),
    ("tldw_chatbook.Widgets.Console.console_auto_speak_consent", "AutoSpeakConsentModal"),
    ("tldw_chatbook.Widgets.Console.console_project_instructions", "ProjectInstructionSetupModal"),
    ("tldw_chatbook.Widgets.Console.console_project_instructions", "ProjectInstructionNoticeModal"),
    ("tldw_chatbook.Widgets.workspace_create_modal", "WorkspaceCreateModal"),
    ("tldw_chatbook.Widgets.project_skills_import_modal", "ProjectSkillsImportModal"),
    ("tldw_chatbook.Widgets.Library.library_note_folder_dialog", "LibraryNoteFolderNameDialog"),
    ("tldw_chatbook.Widgets.Library.library_note_folder_dialog", "LibraryNoteFolderTargetDialog"),
    ("tldw_chatbook.UI.Screens.model_catalog_consent", "ModelCatalogConsentModal"),
]

EXTRA_CONVERSION_TARGETS = [
    ("tldw_chatbook.UI.Console_Modules.right_rail", "ConsoleInspectorRail"),
    ("tldw_chatbook.UI.Screens.trajectory_screen", "TrajectoryScreen"),
    ("tldw_chatbook.UI.Widgets.trajectory_timeline", "TrajectoryTimeline"),
    ("tldw_chatbook.UI.Wizards.first_run_recovery_dialog", "SetupRecoveryDialog"),
    ("tldw_chatbook.Widgets.Console.console_changed_files_section", "ConsoleChangedFilesSection"),
    ("tldw_chatbook.Widgets.Console.console_conversation_inspector", "ConsoleConversationInspector"),
    ("tldw_chatbook.Widgets.Console.console_project_instructions", "ConsoleProjectInstructionContextPanel"),
    ("tldw_chatbook.Widgets.Console.console_project_instructions", "ConsoleProjectInstructionStatusRow"),
    ("tldw_chatbook.Widgets.Console.console_selection_menu", "ConsoleSelectionMenu"),
    ("tldw_chatbook.Widgets.Console.console_transcript", "ConsoleMessageHeader"),
    ("tldw_chatbook.Widgets.Console.console_turn_file_card", "ConsoleTurnFileCard"),
    ("tldw_chatbook.Widgets.Library.library_note_import_canvas", "LibraryNoteImportCanvas"),
    ("tldw_chatbook.Widgets.modal_dismissal", "_BackdropClickShield"),
]


def _register_like_first_mount(app, cls) -> int:
    """Add the sources Textual would add at cls's first mount; return added."""
    added = 0
    for base in cls._css_bases(cls):
        css = base.__dict__.get("DEFAULT_CSS", "")
        if not css:
            continue
        try:
            read_from = (getfile(base), f"{base.__name__}.DEFAULT_CSS")
        except (TypeError, OSError):
            read_from = ("", f"{base.__name__}.DEFAULT_CSS")
        if read_from in app.stylesheet.source:
            continue
        scoped = base.__dict__.get("SCOPED_CSS", True)
        app.stylesheet.add_source(
            css,
            read_from=read_from,
            is_default_css=True,
            tie_breaker=0,
            scope=base._css_type_name if scoped else "",
        )
        added += 1
    return added


#: Textual's parse cache capacity and the repo's soft guard limit -- see
#: Tests/UI/test_widget_css_consolidation.py for the cliff mechanics.
#: NOTE (review, TASK-21115): the live source COUNT is a lower bound on
#: parse-cache entries, not an equality -- the cache key includes
#: tie_breaker (and scope), so a source re-offered at a lowered tie-breaker
#: occupies a fresh cache slot while its old entry ages out. The soft
#: limit's headroom below the cliff absorbs that slack; do not treat
#: "sources == cache entries" as exact when reasoning about the margin.
_PARSE_CACHE_CAPACITY = 64
_SOFT_LIMIT = 56


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tour_plus_modal_opens_stay_under_the_soft_source_limit():
    from importlib import import_module

    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        await asyncio.sleep(2)
        for key in [f"ctrl+{digit}" for digit in "1234567890"] + ["f7", "f8", "f9"]:
            await pilot.press(key)
            await pilot.pause()
            await asyncio.sleep(0.75)
        tour = len(app.stylesheet.source)
        for module, name in MODAL_TARGETS:
            _register_like_first_mount(app, getattr(import_module(module), name))
        after_modals = len(app.stylesheet.source)
        for module, name in EXTRA_CONVERSION_TARGETS:
            _register_like_first_mount(app, getattr(import_module(module), name))
        after_all = len(app.stylesheet.source)
    line = (
        f"TASK-21115 PROBE: tour={tour} "
        f"tour+12modals={after_modals} "
        f"tour+all25targets={after_all} "
        f"(cliff={_PARSE_CACHE_CAPACITY}, soft={_SOFT_LIMIT})"
    )
    print(f"\n{line}")
    assert after_all < _SOFT_LIMIT, (
        f"{after_all} stylesheet sources after the tour plus every "
        f"formerly-DEFAULT_CSS modal/panel -- the 2026-08-22 review's "
        f"feature-rich-session arithmetic is back over the {_SOFT_LIMIT} "
        f"soft limit (cliff at {_PARSE_CACHE_CAPACITY}); {line}"
    )
