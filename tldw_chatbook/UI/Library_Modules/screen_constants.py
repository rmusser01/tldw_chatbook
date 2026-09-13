"""Library screen module-level constants and copy strings.

Moved verbatim out of ``tldw_chatbook/UI/Screens/library_screen.py`` by PR 0a
of the Library screen decomposition
(``.superpowers/sdd/2026-09-01-library-decomposition-foundation``; see
``Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md``).
``library_screen.py`` re-exports every name here so its import surface is
unchanged; later decomposition tasks import directly from this module.

``_INGEST_OPTIONS_CACHE_ATTR`` is deliberately NOT here -- see the
module docstring of ``screen_helpers.py`` for why it stays with the two
ingest-options functions in ``library_screen.py`` instead.
"""
from __future__ import annotations

from ...Library.library_conversation_reader_state import LIBRARY_CONVERSATION_PAGE_SIZE
from ...Library.library_shell_state import (
    LIBRARY_CANVAS_KIND_NOTES_CREATE,
    LIBRARY_ROW_BROWSE_COLLECTIONS,
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_BROWSE_PROMPTS,
    LIBRARY_ROW_BROWSE_SEARCH,
    LIBRARY_ROW_BROWSE_SKILLS,
    LIBRARY_ROW_CREATE_FLASHCARDS,
    LIBRARY_ROW_CREATE_NOTE,
    LIBRARY_ROW_CREATE_PROMPT,
    LIBRARY_ROW_CREATE_QUIZZES,
    LIBRARY_ROW_CREATE_SKILL,
    LIBRARY_ROW_CREATE_STUDY,
    LIBRARY_ROW_INGEST_EXPORT,
    LIBRARY_ROW_INGEST_MEDIA,
)
from ...Prompt_Management.Prompts_Interop import (
    parse_json_prompts_from_content,
    parse_markdown_prompts_from_content,
    parse_txt_prompts_from_content,
    parse_yaml_prompts_from_content,
)
from ...Utils.adaptive_reader_state import AdaptiveReaderLayoutProfile
from ...Widgets.Library import PROMPT_DISCARD_TOOLTIP_BUSY


LIBRARY_SKILLS_IMPORT_WORKER_GROUP = "library_skills_import"


# All destinations retain the shared five-cell collapse controls.
LIBRARY_CONVERSATION_READER_PROFILE = AdaptiveReaderLayoutProfile()
LIBRARY_COLLECTIONS_READER_PROFILE = AdaptiveReaderLayoutProfile(
    work_min_width=48,
    work_comfort_width=56,
)
# task-32127: Notes joined Media on `list_grows` and raised its comfort
# ceiling to 64. At 235 columns the list was pinned at the 40-cell target
# while the Reader held 151 columns of "Select a note to edit it here.",
# which is what clipped the titles, the ages and the delete receipt.
LIBRARY_NOTES_READER_PROFILE = AdaptiveReaderLayoutProfile(
    work_min_width=48,
    list_comfort_width=64,
    list_grows=True,
    # task-32389: task-32217's rule -- an empty work pane hands its columns
    # to the list -- applied below the 64-column floor too. Live at 60x24
    # the notes reader laid out 32/18 with nothing in the work pane.
    list_first_when_empty=True,
)
LIBRARY_FILE_NOTES_READER_PROFILE = AdaptiveReaderLayoutProfile(work_min_width=30)
LIBRARY_PROMPTS_READER_PROFILE = AdaptiveReaderLayoutProfile(work_min_width=48)
LIBRARY_SKILLS_READER_PROFILE = AdaptiveReaderLayoutProfile(
    work_min_width=48,
)
LIBRARY_CONVERSATION_READER_MAX_CHARS = 8000
LIBRARY_SOURCE_PAGE_SIZES = {
    "notes": 100,
    "media": 50,
    "conversations": LIBRARY_CONVERSATION_PAGE_SIZE,
}
LIBRARY_MEDIA_PREVIEW_CACHE_LIMIT = 20
_LIBRARY_PROMPTS_SEARCH_DEBOUNCE_SECONDS = 0.25
# Skills sort modes (Task 3 of the Skills sub-project): "name" (pure
# alphabetical) <-> "status" (needs-review first, then alphabetical) --
# cycled by handle_library_skills_sort. Same "screen-toolbar-cycling
# concern, not pure list-state-building logic" posture as the prompts modes
# above, kept local rather than in library_skills_state.py.
_LIBRARY_SKILLS_SORT_MODES = ("name", "status")


# Toolbar Import… (Task 5): which parser handles which file extension.
# Mirrors ``Prompts_Interop._get_file_type``'s extension map, but writes
# through ``prompt_scope_service``/``LocalPromptService`` per-prompt
# (duplicate-name = skip, never overwrite) rather than
# ``import_prompts_from_files`` -- that helper's own write path
# (``add_or_update_prompt_interop``) hardcodes ``overwrite=True`` with no
# way to opt out, and bypasses the scope service entirely. See
# ``_run_library_prompts_import``.
_LIBRARY_PROMPT_IMPORT_PARSERS = {
    ".json": parse_json_prompts_from_content,
    ".yaml": parse_yaml_prompts_from_content,
    ".yml": parse_yaml_prompts_from_content,
    ".md": parse_markdown_prompts_from_content,
    ".txt": parse_txt_prompts_from_content,
}
_LIBRARY_PROMPTS_IMPORT_WORKER_GROUP = "library-prompts-import"
_LIBRARY_PROMPT_WRITE_WORKER_GROUPS = frozenset(
    {
        "library_prompt_save",
        "library_prompt_history_restore",
        "library_prompt_memberships_apply",
        _LIBRARY_PROMPTS_IMPORT_WORKER_GROUP,
    }
)
_LIBRARY_PROMPT_WRITE_IN_PROGRESS_COPY = PROMPT_DISCARD_TOOLTIP_BUSY
LIBRARY_SERVICE_ERROR_COPY = "Library source services unavailable; retry Library later."
LIBRARY_SERVICE_UNAVAILABLE_COPY = (
    "Library source services are unavailable in this runtime."
)
# task-31632 (critique #5 P1): the source snapshot's own recovery callout --
# the copy for a missed deadline (told apart from the hard failure above),
# the id of the one Retry rendered INSIDE the callout, and the selector the
# landing hub mounts it under.
LIBRARY_SOURCE_TIMEOUT_COPY = "Library sources did not answer"
LIBRARY_SOURCE_RETRY_ID = "library-source-retry"
LIBRARY_SOURCE_FAILURE_SELECTOR = "#library-hub-load-failure"
LIBRARY_EMPTY_COPY = "No local Library content yet."
# F-021: the retired inspector pane's LIBRARY_INSPECTOR_EMPTY_COPY /
# LIBRARY_INSPECTOR_EMPTY_NEXT_ACTION_COPY were deleted -- nothing has
# composed #library-source-inspector since the legacy workbench chrome
# went away, so the (architecture-talk) copy never rendered. The
# user-facing guidance it gestured at lives in the F-013 landing copy
# and the F-010 landing hub.
LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS = 5.0
LIBRARY_ONBOARDING_EVIDENCE_TIMEOUT_SECONDS = LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS
# Navigation composes a FRESH LibraryScreen instance per visit (PR #595
# freeze fix), so a per-instance memo is useless -- the previous visit's
# snapshot is cached on the APP instance instead (see `on_mount` and
# `_refresh_local_source_snapshot`) so a repeat visit within this window
# renders instantly instead of showing the loading placeholder again. The
# cached snapshot is always applied THEN immediately reconciled with a
# fresh background fetch, so staleness is bounded to a single refresh
# cycle regardless of this TTL's length.
LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS = 5.0
# task-2856: how long an "enter/return to a list canvas" entry-focus
# request (``_library_pending_list_entry_focus``) stays armed. A "back to
# list" exit can kick off MULTIPLE sequential background workers, each
# ending in its own recompose (e.g. the skills exit's snapshot refresh
# chains into a trust-posture reload, which does its own LATER
# ``self.refresh(recompose=True)``) -- a single-consume flag loses the
# race against whichever finishes last. ``compose_content`` re-requests
# the focus on every recompose while this window is open instead of
# consuming the flag on first use.
#
# Review round 2: user interaction (ANY key -- ``on_key``; a focus change
# to something other than the armed list's own rows, including mouse
# clicks -- ``on_descendant_focus``) now disarms the request IMMEDIATELY,
# so this window is no longer what protects a user's Tab-away/click from
# being overridden -- it only bounds how long an IDLE, still-armed list
# keeps re-requesting focus across its own chained background workers.
# Measured live (timestamped, two independent runs of the exact skills
# "New skill -> save -> Escape" chain that motivates this at all): the
# gap between the exit's own synchronous recompose and the CHAINED
# trust-posture worker's later one was 249ms and 142ms. This constant is
# ~8-14x that measured worst case, comfortably absorbing a slower disk/
# keyring backend or CI machine while still resolving quickly if a truly
# idle list somehow never settles.
LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS = 2.0
#: task-32228: how often the armed entry focus re-attempts while the list it
#: wants has no rows mounted yet. A coarse poll on purpose -- the only thing
#: it is racing is a canvas-level recompose, and every tick is dropped by the
#: arm's own generation check once the user takes control.
LIBRARY_LIST_ENTRY_FOCUS_RETRY_SECONDS = 0.05
LIBRARY_NOTES_AUTOSAVE_SECONDS = 2.0
LIBRARY_NOTE_CONTENT_MAX_CHARS = 2_000_000
# The literal title a just-created "Blank note" row is seeded with (LIB-14,
# task-4021). The editor presents it placeholder-only (empty Input,
# "Untitled" placeholder -- see ``_notes_state.pending_blank_gc_id``), and
# the untouched-blank GC gate (``_flush_library_note_save``) treats a
# snapshot title equal to this literal as blank too -- both call sites and
# the create seed in ``handle_library_notes_create_blank`` must agree on
# the one constant rather than three independently-typed string literals.
LIBRARY_NOTE_BLANK_SEED_TITLE = "Untitled"


# Prompt editor body fields (details/system/user) have no dedicated cap of
# their own -- reuses the note body's generous ceiling rather than inventing
# a second magic number for the same "large text field" concern.
LIBRARY_PROMPT_TEXT_MAX_CHARS = LIBRARY_NOTE_CONTENT_MAX_CHARS
# Exact outcome copy for the prompt editor's #library-prompt-save-status
# line, keyed by `classify_prompt_save_error`'s return value. "conflict" is
# deliberately absent -- both the pre-write staleness check AND a
# ConflictError raised by the write itself (a race the pre-check cannot
# see) route into the conflict banner instead (see `_save_library_prompt`),
# never this status line.
LIBRARY_PROMPT_SAVE_STATUS_COPY = {
    "ok": "Saved.",
    "name-in-use": "Name already in use — pick another or open the existing prompt.",
    "soft-deleted-name": "A deleted prompt holds this name — restore it or choose another.",
    "error": "Couldn't save this prompt. Try again.",
}
# Skill editor's text fields (description/allowed-tools/body) have no
# dedicated cap of their own -- reuses the note body's generous ceiling,
# same reasoning as ``LIBRARY_PROMPT_TEXT_MAX_CHARS`` above.
LIBRARY_SKILL_TEXT_MAX_CHARS = LIBRARY_NOTE_CONTENT_MAX_CHARS
LIBRARY_PROMPT_DIRTY_VETO_COPY = (
    "Unsaved Prompt changes — Save or Discard changes first."
)
# task-32393: the footer's Escape chip while that veto is in force. "esc back to
# list" is a promise the key will refuse to keep once the editor is dirty, which
# is the same lie task-31271/31272 closed at the other Library seams -- so the
# chip names the blocker instead, in the veto toast's own words.
LIBRARY_PROMPT_DIRTY_ESCAPE_CHIP = "save or discard first"
# PR #2655 (Qodo finding 1): the same rule for the OTHER state that seam
# refuses in -- ``_exit_library_prompt_editor_guarded`` returns False on an
# in-flight write before it ever reads ``dirty``, so a clean prompt deletion
# left the chip promising an exit the key would only answer with the busy
# warning. Takes precedence over the dirty chip, matching that check order.
LIBRARY_PROMPT_BUSY_ESCAPE_CHIP = "busy, try again"
# Exact outcome copy for the skill editor's #library-skill-save-status line,
# keyed by ``classify_skill_save_error``'s return value. "version-conflict"
# is deliberately absent -- it routes into the conflict banner instead (see
# ``_save_library_skill``), never this status line.
# task-449: toast shown whenever a dirty skill edit vetoes an exit (Back,
# skill-row switch, rail-row switch) -- the veto itself is silent widget
# state, so without this the blocked click looks like a dead button.
LIBRARY_SKILL_DIRTY_VETO_COPY = "Unsaved skill changes — Save or Discard changes first."
# task-414: specific approve-failure copy for the service's
# ``ValueError("snapshot_mismatch")`` -- the files changed between capture
# and approve, and the service has already discarded the review.
LIBRARY_SKILL_TRUST_MISMATCH_COPY = (
    "Skill files changed after the review was captured, so it was discarded. "
    "Press Review changes again, then Approve."
)
LIBRARY_SKILL_SAVE_STATUS_COPY = {
    "ok": "Saved.",
    "exists": "A skill with this name already exists.",
    "invalid-name": "Skill name must use lowercase letters, numbers, and hyphens.",
    "trust-blocked": "This skill is blocked by trust review — approve it in the trust panel before saving.",
    "error": "Couldn't save this skill. Try again.",
}
LIBRARY_COLLECTION_SYNC_CONFLICT_LIMIT = 200
LIBRARY_HANDOFF_LABEL_PREFIX = "Console/RAG handoff: "
LIBRARY_WORKSPACE_SOURCE_COLUMN_WIDTH = 30
LIBRARY_WORKSPACE_SCOPE_COLUMN_WIDTH = 18
LIBRARY_WORKSPACE_VISIBLE_COLUMN_WIDTH = 7
LIBRARY_WORKSPACE_CONTEXT_COLUMN_WIDTH = 11
LIBRARY_HUB_RECENT_LABEL_WIDTH = 32
LIBRARY_MEDIA_HANDOFF_EXCERPT_CHARS = 500
# `_refresh_library_rag_results_widgets` tears down every direct child of
# `#library-rag-results` NOT in this set, then remounts fresh ones from
# `library_rag_results_body_children` -- the same function `compose()`
# uses, so the two paths cannot drift. Task 12/RAG-36 wrapped each row's
# several flat sibling widgets into ONE `.library-rag-result-card`
# container per row; that reduces the child COUNT under
# `#library-rag-results` but does not change this set's membership, since
# every row-level id (old flat widgets or the new card) was already being
# removed and remounted here -- only the always-kept heading is listed.
LIBRARY_RAG_RESULTS_STATIC_WIDGET_IDS = frozenset({"library-rag-results-heading"})
LIBRARY_NOTES_COMPACT_BREAKPOINT = 120
LIBRARY_INGEST_RAIL_COLLAPSE_BREAKPOINT = 100
# TASK-23025: the adaptive reader shells, all constructed exclusively in
# ``compose_content`` -- the basis for the one-probe-per-recompose negative
# cache in ``_library_adaptive_reader_shell_active``.
_LIBRARY_READER_SHELL_SELECTOR = (
    # Phase C: Media and Notes now share ONE resident shell
    # (``#library-browse-reader-shell``), so this union lists five ids for
    # six routes.
    "#library-browse-reader-shell, "
    "#library-collections-reader-shell, "
    "#library-conversations-reader-shell, "
    "#library-prompts-reader-shell, "
    "#library-skills-reader-shell"
)
LIBRARY_NOTES_SOURCE_DATABASE = "database"
LIBRARY_NOTES_SOURCE_FILES = "files"
# task-32050: a note load that never returns used to leave the canvas on
# "Loading note…" for the rest of the session. Past this deadline the load
# is abandoned into the existing failed state, which carries Retry.
LIBRARY_NOTE_LOAD_DEADLINE_SECONDS = 3.0
LIBRARY_NOTE_LOAD_TIMEOUT_COPY = (
    "Unable to load note — timed out after 3 s. Press Retry."
)
LIBRARY_CANVAS_KIND_NOTES = "notes"
LIBRARY_NOTES_SOURCE_STRIP_CANVAS_KINDS = frozenset(
    {LIBRARY_CANVAS_KIND_NOTES, LIBRARY_CANVAS_KIND_NOTES_CREATE}
)

#: The Notes views that are a whole TASK rather than a second reading
#: surface (task-32259 AC#2). While one of these is in hand the Items list
#: it is not about closes to its grip, so the task owns the pane width --
#: the notes list is not a companion to an import review.
LIBRARY_NOTES_FULL_CANVAS_VIEWS = frozenset({"import", "lasting_add", "lasting_roots"})

#: The rail rows that land on the Database-Notes route. Named (phase-C task
#: 2.5) because the rail-switch handler has to ask "is this press LEAVING
#: Notes?" before it decides whether repainting the Notes canvas is work
#: anybody will ever see.
LIBRARY_NOTES_RAIL_ROWS = frozenset({LIBRARY_ROW_BROWSE_NOTES, LIBRARY_ROW_CREATE_NOTE})


# PR-3 Task 4: the retrieval outcomes phase two runs on. `ready` is the
# ordinary case; `empty` is answered too -- honestly, and without a provider
# call (`generate_library_rag_answer` refuses to hand a model an evidence
# block with nothing citable). `blocked`/`failed` are excluded on purpose:
# retrieval did not run, so there is nothing to be grounded in and the
# recovery copy those statuses already render is the honest answer.
LIBRARY_RAG_ANSWERABLE_RETRIEVAL_STATUSES = frozenset({"ready", "empty"})


LIBRARY_STUDY_HANDOFF_MODES = {
    "study": {
        # "header" mirrors the rail row title that opens this canvas
        # (LibraryRailRow "Study decks", library_shell_state.py) so the
        # canvas doesn't restate the mode name a second, differently-worded
        # way (L3b Task 8/9 follow-up: UX wave C/D, handoff copy
        # consolidation).
        "header": "Study decks",
        "action_label": "Study Dashboard",
        # UX wave L2: the button reads as a verb ("Continue in Study")
        # instead of restating the destination's own name -- action_label
        # still backs the header/purpose/recovery copy below.
        "button_label": "Continue in Study",
        "purpose": "Plan study decks from Library sources.",
    },
    "flashcards": {
        "header": "Flashcards",
        "action_label": "Flashcards",
        "button_label": "Continue in Study",
        "purpose": "Generate or review cards from Library sources.",
    },
    "quizzes": {
        "header": "Quizzes",
        "action_label": "Quizzes",
        "button_label": "Continue in Study",
        "purpose": "Generate or resume quizzes from Library sources.",
    },
}

# Single shared ownership line for all three handoff canvases: Library only
# prepares source context, Study owns everything downstream of "open".
# task-32069: the rail spent a second row per handoff row repeating "see
# what carries over" -- six rows for three destinations. The promise belongs
# on the staging canvas that keeps it, so it is stated here instead.
LIBRARY_STUDY_HANDOFF_OWNERSHIP_COPY = (
    "This page shows what carries over; generation and review run in Study."
)

# How many carried-forward source titles the handoff canvas names before
# collapsing the rest into an "and N more" count.
LIBRARY_STUDY_HANDOFF_TITLES_CAP = 3


# Maps a Library navigation-context ``mode`` value to the shell rail row
# that selects that canvas -- covers exactly the mode values nav-context
# callers emit/support today (L3b Task 8 audit): ``conversations`` (Personas'
# conversations controller), ``search`` and ``collections`` (both directly
# tested contracts of ``apply_navigation_context``, though no live emitter
# currently sends them), and ``prompts`` (the retired Personas "prompts" mode
# chip's legacy route alias -- see ``screen_registry``'s ``_SCREEN_ALIASES``
# and ``shell_destinations``, Task 7). ``skills`` (Skills sub-project Task 1)
# has no live emitter yet either -- same forward-compat posture as
# ``search``/``collections`` -- added so a future Skills deep link has
# somewhere to land without another table edit. ``notes`` is handled as its
# own dedicated branch in ``_apply_navigation_context_state`` below
# (``open_notes_workspace``'s route), not through this table. ``media``
# (task-2851: the retired standalone Media Library screen's legacy route
# alias -- mirrors "search"/"prompts"/"skills" above) lands on this same
# canvas's browse row -- unlike those, its own selected item still round-
# trips through ``open_source_type``/``open_source_id`` above rather than a
# mode-table entry, so this only owns landing on the list. ``study`` (task-
# 2854: the Study screen's Escape binding lands back here) is a "handoff"
# row like ``flashcards``/``quizzes`` -- unlike the plain "canvas" rows
# above, selecting it shows the staging canvas ("Continue in Study"), not a
# browse list -- but ``_select_library_rail_row_after_source_admission``
# treats every row_id identically regardless of target_kind, so nothing
# about being a handoff row stops it from being a valid nav-context landing
# spot. ``flashcards``/``quizzes`` stay out: nothing emits those modes yet
# (a Library-origin Escape always returns to the shared "Study decks" row --
# see ``StudyScreen.action_study_back``), so adding them now would be
# speculative, same posture as the other forward-compat-only entries below.
# Any other mode value, including the retired ``sources``/``workspaces``/
# ``import-export`` values, degrades quietly, unchanged from before this
# table existed.
LIBRARY_NAV_MODE_TO_ROW_ID = {
    "conversations": LIBRARY_ROW_BROWSE_CONVERSATIONS,
    "collections": LIBRARY_ROW_BROWSE_COLLECTIONS,
    "search": LIBRARY_ROW_BROWSE_SEARCH,
    "prompts": LIBRARY_ROW_BROWSE_PROMPTS,
    "skills": LIBRARY_ROW_BROWSE_SKILLS,
    "media": LIBRARY_ROW_BROWSE_MEDIA,
    "study": LIBRARY_ROW_CREATE_STUDY,
}

#: task-4023 AC#7: the three Study staging (handoff) rows -- the surfaces
#: whose Escape returns to the hub landing (they had no back path at all).
#: Row ids match ``library_shell_state.py``'s study_rows.
LIBRARY_STUDY_HANDOFF_ROW_IDS = (
    LIBRARY_ROW_CREATE_STUDY,
    LIBRARY_ROW_CREATE_FLASHCARDS,
    LIBRARY_ROW_CREATE_QUIZZES,
)


# --- task-2856: Library keyboard story -------------------------------------
#
# Every list canvas (Media/Notes/Prompts/Skills) renders its rows as plain
# Button widgets in a Vertical container, not a Textual ListView -- so Up/
# Down movement between them has to be wired by hand. The four row CSS
# classes are the seam: entering a list canvas focuses the first one
# (``_focus_library_list_entry``/``_LIBRARY_LIST_ROW_CLASS_BY_ROW_ID``) and
# Up/Down move DOM focus between siblings sharing one of these classes
# (``_move_library_list_row_focus``). A MODULE-LEVEL function (not a method)
# for the same reason ``_apply_library_row_toggle`` below is one: it is unit
# tested against a minimal Pilot host without constructing a full
# ``LibraryScreen``.
_LIBRARY_LIST_ROW_CLASSES = (
    "library-media-row",
    "library-notes-row",
    "library-prompt-row",
    "library-skill-row",
    # task-32052 AC#2: the New-note canvas's Blank note + template rows are
    # a stacked list of full-width rows styled after ``library-notes-row``,
    # so Up/Down must walk them like any other Library list. The "From a
    # template" heading between them is a ``Static``, which this helper's
    # own class filter already skips.
    "library-notes-create-row",
    # task-32228 (critique #9 fix round 1): Conversations was the one browse
    # list left out of both halves of this seam. Its rows are the same
    # full-width ``library_row_button`` the four above use
    # (``library_conversations_canvas.py``), so Up/Down now walks them too.
    "library-conversation-row",
)

#: task-32225 (re-review N4): the canvas filters "/" prefers over the rail's
#: global search, keyed by rail-row id. Named once so the ``on_key`` handler
#: that FOCUSES one and the footer predicate that decides whether to advertise
#: the key read the same two selectors. The handler stays the authority on
#: order; this is only its lookup table.
_LIBRARY_SLASH_CANVAS_FILTERS = {
    LIBRARY_ROW_BROWSE_MEDIA: "#library-media-filter",
    LIBRARY_ROW_BROWSE_PROMPTS: "#library-prompts-filter",
}

_LIBRARY_LIST_ROW_CLASS_BY_ROW_ID = {
    LIBRARY_ROW_BROWSE_MEDIA: "library-media-row",
    LIBRARY_ROW_BROWSE_NOTES: "library-notes-row",
    LIBRARY_ROW_BROWSE_PROMPTS: "library-prompt-row",
    LIBRARY_ROW_BROWSE_SKILLS: "library-skill-row",
    # ``_enter_library_prompt_create_editor``/``_enter_library_skill_create_
    # editor`` (Create rail row -> editor) never reassign
    # ``_library_selected_row_id`` away from these CREATE_* ids (mirroring
    # ``_library_skill_editor_active``'s own dual-row-id gate) -- so
    # exiting that editor back to the list still reads CREATE_PROMPT/
    # CREATE_SKILL here, not BROWSE_PROMPTS/BROWSE_SKILLS. Omitting these
    # left "New skill -> save -> Escape" focusing nothing (reproduced
    # live: ``row_class`` resolved to ``None`` for ``LIBRARY_ROW_CREATE_
    # SKILL``).
    LIBRARY_ROW_CREATE_PROMPT: "library-prompt-row",
    LIBRARY_ROW_CREATE_SKILL: "library-skill-row",
    # task-32228: see above. Arriving with focus on the first row is what
    # makes this canvas's Escape hop live, and therefore what lets its footer
    # advertise the key honestly.
    LIBRARY_ROW_BROWSE_CONVERSATIONS: "library-conversation-row",
}


#: task-4023 AC#4 (RC-10): human names for the F1 panel's surface-qualified
#: title, keyed by rail-row id. Surface identity only -- the shortcut SET
#: still comes solely from ``_library_footer_shortcuts_for_current_state``.
_LIBRARY_HELP_SURFACE_LABELS: dict[str, str] = {
    LIBRARY_ROW_BROWSE_MEDIA: "Media",
    LIBRARY_ROW_BROWSE_CONVERSATIONS: "Conversations",
    LIBRARY_ROW_BROWSE_NOTES: "Notes",
    LIBRARY_ROW_BROWSE_PROMPTS: "Prompts",
    LIBRARY_ROW_BROWSE_SKILLS: "Skills",
    LIBRARY_ROW_BROWSE_COLLECTIONS: "Collections",
    LIBRARY_ROW_BROWSE_SEARCH: "Search / RAG",
    LIBRARY_ROW_INGEST_MEDIA: "Import",
    LIBRARY_ROW_INGEST_EXPORT: "Export",
    LIBRARY_ROW_CREATE_NOTE: "New note",
    LIBRARY_ROW_CREATE_PROMPT: "New prompt",
    LIBRARY_ROW_CREATE_SKILL: "New skill",
}


# --- relocated from ``library_screen.py`` by wave-7 task 2 (media series
# --- 2/3): four constants moved bodies read as bare module globals, moved
# --- verbatim so the single-source guarantee each of them exists to give is
# --- preserved rather than re-spelled in a second file. ``library_screen.py``
# --- imports them back, so ``library_screen_module._ANALYZE_ORIGIN_MEDIA``
# --- and ``..._IMPORT`` still resolve for their two pinning assertions in
# --- ``Tests/UI/test_library_ingest_analyze_skipped.py``.

#: Qodo on #2386: the two ``_library_media_view`` values this branch's
#: surface helpers compare against. Introduced with those helpers; the
#: file's older sites still carry the literals.
_MEDIA_VIEW_LIST = "list"
_MEDIA_VIEW_VIEWER = "viewer"
# task-28007 (Qodo review round, PR #2400 #1): the two surfaces a bulk-
# Analyze run can start from. Named once so the init default
# (``_library_media_analyze_origin``), its assignment in
# ``_start_library_media_analyze``, and its two readers (the unmount
# notice, the receipt-fields guard) cannot drift apart from one another.
_ANALYZE_ORIGIN_MEDIA = "media"
_ANALYZE_ORIGIN_IMPORT = "import"
