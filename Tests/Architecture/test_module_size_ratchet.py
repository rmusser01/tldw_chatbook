"""A size ratchet for the largest ungoverned modules — stop them growing.

**Why this test exists.** The core-runtime code review (2026-09-17,
`Docs/QA/core-code-review-2026-09-17/report.md`) found that several of the
repository's largest modules had no size ratchet row at all, across
directories that neither `test_screen_size_ratchet.py` (screens) nor
`test_library_modules_size_ratchet.py` (`Library_Modules/*_controller.py`)
covers. Without a row, nothing stops them growing — the same silent creep
those two files were written to catch, on the modules that need it most.

**This is a ratchet, not a limit.** Each budget below is pinned at the
module's exact current line count (`len(path.read_text().splitlines())`,
the expression `_measure` uses). The numbers may only ever go DOWN. When
you shrink one of these files, lower its row to the new measurement in the
same commit. If you are here because CI failed, the fix is to put your new
code somewhere else — a controller, a widget, a helper module — never to
raise the number, which re-opens the hole this test exists to close.

**Scope.** These are hand-picked god modules, not a directory family, so
(unlike the Library controller ratchet) there is no glob that auto-adds new
files. `settings_screen.py` joined on 2026-10-05 (TASK-33007.8, closing
task-31202), once the model-config redesign's Phase 7 had moved the
Providers & Models card into `UI/Settings_Modules/`; the rest of its split
is task-1378.

First recorded 2026-09-19 by core-review TASK-32809.2, each row at its exact
measured size as of `origin/dev`.

**Second pass, 2026-09-21 (tier-2 review, task-32901).** TASK-32809.2's
hand-picked list missed four modules LARGER than three of the seven rows it
did pick — the tier-2 review found them in four separate slices, each slice
independently reporting "god module with no ratchet row":

* `UI/Wizards/FirstRunSetupWizard.py` (10,404) — the largest module in the
  repo with no row at all (S21). `backlog/docs/size-decomposition-candidates-
  2026-09-18.md` says "Size ratchets now guard all of these" and does not
  list `UI/Wizards/` anywhere.
* `UI/Screens/watchlists_collections_screen.py` (14,319) — one class, 392
  methods, roughly double the method count of the biggest budgeted
  non-`chat`/`library` class (S18). Neither `test_screen_size_ratchet.py`
  (which holds only `chat_screen` and `library_screen`) nor this file had it.
* `UI/Screens/llm_screen.py` (5,180) and `UI/Screens/change_review_screen.py`
  (4,967) — the other two S18 named.

They are budgeted HERE rather than in `test_screen_size_ratchet.py` because
that file's rows also pin a method count per named class, which needs a
decomposition plan to be meaningful; `personas_screen.py` is the standing
precedent for a screen living in this file. No split is proposed by adding a
row — `backlog/docs/library-decomposition-recipe.md` governs any split.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from .test_library_modules_size_ratchet import _BUDGETS as _LIBRARY_BUDGETS
from .test_screen_size_ratchet import _BUDGETS as _SCREEN_BUDGETS

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: path -> max line count. LOWER these when a module shrinks. Never raise
#: them to silence a failure — see the module docstring.
_BUDGETS: dict[str, int] = {
    # TASK-33011 decomposition: lowered as each extraction PR lands
    # (entry tail -> app_entry.py: 21,234 measured -> 20,484; destinations
    # D/K2/K3 -> app_destinations.py: 20,484 -> 19,658; task-33081 on dev
    # +24 -> 19,682; dev growth (theme tail, Dreams wiring) +191 -> 19,873;
    # Library ingest queue -> app_ingest_queue.py: 19,873 -> 14,930; dev
    # growth (theme final wave, SSH sessions) +42 -> 14,972; service
    # composition C/E/O -> app_service_wiring.py: 14,972 -> 11,506; speech
    # N/U/U2 -> app_speech.py: 11,506 -> 10,601; lifecycle, shutdown and
    # quit flow -> app_lifecycle.py: 10,601 -> 8,524; screen navigation ->
    # app_navigation.py: 8,524 -> 7,467; command-palette providers ->
    # app_command_providers.py: 7,467 -> 6,403; per-feature glue ->
    # app_feature_glue.py: 6,403 -> 5,712).
    # 2026-10-19 (TASK-33620.15.5, owner decision: raise the row rather than
    # contort code): +4 lines for the idle pre-import of the Console's
    # first-send modules, which must run in the post-UI-ready pass itself.
    # Re-measured 2026-10-20 on origin/dev 0fef489ece (app.py 5,707 there):
    # 5,710 -> 5,711.
    # 2026-10-23, TASK-34000.71 fix round 1 (owner decision: a fix over a
    # size ratchet raises the row): 5711 -> 5712 (+1) -- the
    # ``_notes_sync_runtime_lazy_start_task`` slot beside the start-task
    # slot, so the Library's bounded settle wait can tell the task it
    # scheduled from boot's own (M-1).
    "tldw_chatbook/app.py": 5712,
    # TASK-34406.5: unchanged startup/footer wiring, born governed.
    "tldw_chatbook/app_startup.py": 176,
    # TASK-34406.7: app-owned served control and relay bodies, born governed.
    "tldw_chatbook/app_served.py": 125,
    # TASK-33011: the Library ingest queue moved verbatim out of app.py. At
    # 4,999 lines it is larger than two rows below, so it is born governed.
    # 2026-10-03 (PR #2993): TASK-20973's provenance block added 16 lines;
    # the signature probe moved beside the poller's other pure helpers in
    # Library/server_ingest_reconcile.py: 5,015 -> 4,985.
    "tldw_chatbook/app_ingest_queue.py": 4985,
    # TASK-33011: TldwCli's service composition moved verbatim out of app.py.
    # It was governed there; without its own row, wiring code could regrow in
    # the mixin unchecked.
    # 2026-10-23, TASK-34000.71 (owner decision: a fix over a size ratchet
    # raises the row, never contorts the code): 3573 -> 3633 (+60) --
    # ``ensure_notes_sync_runtime_started``, the Library side's one entry
    # for the lazy non-forced start through boot's own start-task
    # bookkeeping (its docstring carries the deadlock it breaks), plus the
    # done-callback's no-screen-stack guard and two imports.
    # 2026-10-23, TASK-34000.71 fix round 1 (same owner decision): 3633 ->
    # 3690 (+57) -- the entry point re-arms a failed or cancelled start
    # while the owner still reports "starting" (M-2), tolerates a sync or
    # Future-returning ``start`` (M-3), records the lazy task (M-1); three
    # module helpers carry those predicates.
    "tldw_chatbook/app_service_wiring.py": 3690,
    # TASK-33011: TldwCli's lifecycle, shutdown and quit flow moved verbatim
    # out of app.py (LifecycleMixin); governed there, so governed here.
    # TASK-33621.13: 2,140 -> 2,134 (keep-alive dead-pump handling moved
    # into app_keep_alive.py).
    "tldw_chatbook/app_lifecycle.py": 2134,
    # TASK-33011: TldwCli's screen navigation moved verbatim out of app.py
    # (NavigationMixin); governed there, so governed here.
    # TASK-34000.27 (2026-10-09): 1110 -> 1195 (+85), owner decision
    # 2026-10-04 (raise the row to the measured value rather than move
    # unrelated code to fit): the one nav-bar rollback helper
    # (`_restore_nav_bar_highlight`, extracted from the crash path) is
    # now called from every refusing branch of the locked navigation
    # body, and a flush that accepts `destination` is told the clicked
    # destination's label (`_navigation_destination_label`,
    # `_flush_accepts_destination`). Review 1: 1195 -> 1208 (+13), the
    # admission-veto branch says the honest generic ("Couldn't leave this
    # screen yet; try again in a moment.") -- no admission hook speaks on
    # its own.
    "tldw_chatbook/app_navigation.py": 1208,
    # TASK-33011: the command-palette providers moved verbatim out of app.py;
    # governed there, so governed here.
    "tldw_chatbook/app_command_providers.py": 1123,
    # TASK-33011: TldwCli's per-feature glue moved verbatim out of app.py
    # (FeatureGlueMixin); governed there, so governed here.
    "tldw_chatbook/app_feature_glue.py": 742,
    # TASK-35190: the rebaser's endpoint choice moved to
    # console_settings_apply.rebased_endpoint: 29,182 -> 29,150.
    # Rebased onto dev 0fef489ece, whose TASK-35444 Console refresh deferral is +1: 29,151.
    # TASK-35992: frozen Library preparation delegates to console_library_preparation.
    # 2026-10-22 (TASK-33621.20, owner decision 2026-10-04: raise the row
    # rather than squeeze unrelated code): +12. Rebased onto dev 7700c63a68
    # (29,043), the Library-pause fixes measure 29,055: dev's TASK-35992
    # extraction already moved out the Library preparation body that these
    # fixes used to shrink here, so their stop/shelf/card wiring now shows.
    # TASK-33621.51 (2026-10-29): the Library preparation record builders
    # moved to console_library_preparation (preparation_contribution,
    # bypass_contribution): 29,055 -> 29,038 on dev ed7481a30c.
    "tldw_chatbook/Chat/console_chat_controller.py": 29038,
    # ADR-220: existing interrupt and compaction owners gain explicit coordination.
    "tldw_chatbook/Chat/console_interrupt_rounds.py": 6307,
    "tldw_chatbook/Chat/console_context_compaction.py": 4185,
    "tldw_chatbook/Chat/console_chat_store.py": 22338,
    # TASK-33622.14: the aggregate Roleplay draft guard moved to
    # UI/Persona_Modules/roleplay_draft_guard.py (dev had grown to 16,533,
    # over this row; the move brings it to 16,397).
    # 2026-10-03 (PR #2993, owner decision): +128 is formatter reflow only
    # (TASK-26000 series, Ruff 0.15.22; the file is AST-identical to the
    # 16,397-line version), so the row is re-measured, not grown.
    # 2026-10-04 (TASK-34400, owner decision): +3 for the crash fix that stops
    # names shaped like markup from exiting the app (the escape_markup import,
    # its comment, and a wrapped _notify docstring line). The owner ruled to
    # raise the row rather than squeeze unrelated code to fit.
    # 2026-10-09 owner decision (TASK-33910.2, 'Raise it in B1'): dev's #3045
    # grew personas_screen.py from 16,528 to 16,567 lines; B1 brings it to
    # 16,564.
    # TASK-36097 extracts Persona completion without raising the ceiling.
    # Roleplay frame B2a (TASK-33910.3), owner decision 2026-10-19 ("expand
    # limits, never contort code"), applied 2026-10-22 and 2026-10-23: B2a
    # grows personas_screen.py from 16,561 lines (dev 7700c63a68) to 16,885;
    # the row rises in each commit that grows it.
    # TASK-36454..36458: narrow completion helpers lower the combined owner.
    # TASK-33910.34 fix round 1, same decision, applied 2026-10-24: a failed
    # or unstartable window read drops the End or load-through it was for
    # (one helper and five calls) and the replay moves into the list; +8
    # lines on the task's own net -3, so the row rises from dev's measured
    # 16,878 (dev 96cae7eec0) to 16,883.
    # TASK-36932 moves the bounded Lore save operation beside existing completions.
    # Rebased onto dev386 with the search/Save publication correction:
    # 16,870 combined, still 13 below current dev's 16,883 ceiling.
    "tldw_chatbook/UI/Screens/personas_screen.py": 16859,
    # Roleplay frame B1: recipient ceiling for the moved header code (spec 5.4).
    # Roleplay frame B1 (TASK-33910.2), owner decision 2026-10-04 ("expand the
    # limits"), applied 2026-10-09: the final-review fix that puts a name and a
    # server label on one header row adds 15 lines; the row rises 369 -> 384,
    # never squeezed code.
    # Roleplay frame B2a (TASK-33910.3), owner decision 2026-10-19 ("expand
    # limits, never contort code"), applied 2026-10-22: the items-list state
    # (prompt builder, density, count text, marks-bar and delete-modal copy,
    # window rules; spec 5.11's B2a row) adds +536 lines (384 -> 920); the
    # row rises with it, never squeezed code. B2a Task 5, same decision and
    # day: the marks bar's narrow forms (whole actions drop, none is cut)
    # add +8 (920 -> 928), and the Task 5 review's fix (the count yields
    # before Delete, so Delete never drops) adds +3 (928 -> 931).
    # TASK-33910.34, owner decision ("expand limits, never contort code", as
    # ruled for B2a on 2026-10-19), applied 2026-10-24: the items list's
    # window state gets one owner (ListWindowState, WindowIntent, same_list)
    # and adds +178 lines (931 -> 1,109). Its fix round 1 (same day) adds
    # ``read_failed``: +7 (1,109 -> 1,116). TASK-33910.35, same decision,
    # applied 2026-10-25: "land on the first match" is one more WindowIntent
    # reason (owe_first, landing_owed, take_first); +29 (1,116 -> 1,145).
    # Its fix round 1 (same day) adds landed_elsewhere (a landing owed on a
    # superseded non-character list is dropped); +16 (1,145 -> 1,161).
    "tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py": 855,
    # Roleplay frame B2a (TASK-33910.3): recipient ceiling (spec 5.4 item 4,
    # 5.10) for the items list that replaced the ListView and its pager,
    # pinned at its measured size (727 on dev -> 1,223; owner decision
    # 2026-10-19, "expand limits, never contort code"; applied 2026-10-23).
    # The final review's fixes (same decision and day) raise it to 1,234.
    # TASK-33910.34, same decision, applied 2026-10-24: the list reads its
    # window state from the shared owner, and each cursor key or click forgets
    # an owed End or load-through; +44 lines (1,234 -> 1,278). Its fix round
    # 1 (same day): only a selection of another row or Enter is a newer move,
    # and the list asks again for a replayed move; +17 (1,278 -> 1,295).
    # TASK-33910.35, same decision, applied 2026-10-25: Enter on the list is
    # inert while the first-match landing is owed, and the pane takes the
    # landing (land_on_first); +12 (1,295 -> 1,307). Fix round 1 (same day)
    # deletes the unused focus_first_result helper: -8 (1,307 -> 1,299).
    "tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py": 1297,
    # TASK-33007.8 (closes task-31202): pinned at its measured size after
    # model-config Phase 7 moved the Providers & Models card out (34,179 at
    # the branch base 9b28ce1479 -> 33,607). task-1378 owns the rest of the
    # split; its new code goes in UI/Settings_Modules/, never in this row.
    # The Phase 7 final review's fixes, still inside the PR that adds this
    # row, re-pin it at the PR head's measured size (R4): 33,638. The
    # full-screen capture fixes (2026-10-09) put their new code in the card
    # module and re-pin it at 33,647. Moving the Endpoint guide there too
    # (2026-10-09, it says required as the row does) lowers it to 33,643.
    # Integrating the capture review's items 9-12 (Console Behavior's rows,
    # one-row Context window, its unknown help naming the assumed size)
    # re-pins it at the integrated head's measured 33,645; the checkpoint
    # review's result-line fix (SettingsResultLine) re-pins it at 33,661.
    # Rebased onto dev (2026-10-09), which added its quit-time session summary
    # helpers here (+80 on dev): re-pinned at the rebased head's 33,712, still
    # under dev's 34,598 for the file; the Provider list's prefix-first
    # highlight makes it 33,724; dev's #3056 session-summary deferrals add 8
    # (rebased again): 33,732. The review of #3057 (2026-10-11, every finding
    # fixed by owner ruling) re-pins it at the integrated head's 33,881: the
    # quit summary's single-writer save and its Inspector guide (C8), the
    # staged credential source behind the Key check row, the deep link's
    # first edit (C1) and Save selected's own worker group and lock (C9).
    # Fix round 1 of that review ends one line under it, at 33,880; fix
    # round 2 sets the quit summary's field to the duration a write sends
    # (+1): 33,881; a linked pair reads "edited *" only beside an edit Save
    # writes it with (+4): 33,885; the Test result blames the Env var field
    # only for a name typed there that is no name (+8): 33,893; fix round 3
    # compares a quit-summary change with the write in flight (+2): 33,895.
    # The capture fix wave (2026-10-12): a draft category at rest says "No
    # changes" in its State banner, not "Draft" (+4): 33,899; a revert's
    # one report is its toast, and the next edit clears a save's report
    # (+6): 33,905; discovery names the provider, and its empty state
    # is one line, not two (-5): 33,900; Clear's rule moves to the card
    # (-9): 33,891; Temperature's range is the provider's own in its
    # placeholder, its guide and Save (+8): 33,899; Providers & Models'
    # Inspector prints no "Owns" list of raw keys (+4): 33,903.
    # Capture fix A2 (2026-10-12): Catalog refresh keeps its receipt for a
    # write and says nothing at rest (+4): 33,907; its focus-help line skips
    # the nav bar's tooltips (+3): 33,910; Console Behavior's Status row and
    # Chat images become a Select and a Checkbox, the summary prompt's
    # button gets a labelled row (+2): 33,912. Capture fix round 1: those
    # two instant-apply rows' help leads with "Applies immediately." (+0).
    # Capture round 2 A1: the Inspector's Save row says how edits save,
    # never "staged" (+1): 33,913; the Inspector's draft line names the
    # unsaved fields, one list behind it and the count (+23): 33,936; the
    # provider list's "Enter provider ID" has no heading of its own, and
    # the guide calls it that, not "Manual" (+4): 33,940; a discovery's
    # one report is its toast, and its card line hides when empty (+1):
    # 33,941; the Inspector scrolls no further than a focused field's guide
    # needs and stops on a heading, after a revert and a resize too, and its
    # fold row says when the rest is above (-8): 33,933; the unsaved names
    # move from the pinned draft line to the note (-9): 33,924.
    # Capture fix A2 round 2 (2026-10-13): Console Behavior's Inspector
    # Scope row names the card's sections, not config sections (-1): 33,911;
    # Reasoning replay override hides its controls until Console names a
    # local model (+8): 33,919; its checkboxes say their state in the glyph,
    # a check or a blank (StateCheckbox, +1): 33,920; Permission summaries'
    # Provider is a list of providers, its options built in
    # settings_field_rows.py (+5): 33,925.
    # Capture fix 2, group C: quitting with a staged change asks first; the
    # prompt lives in Settings_Modules/settings_quit_guard.py (+5): 33,917.
    # Integrated capture fix 2 (2026-10-13): re-measured after the pick: 33,942.
    # Capture fix 2 round 1: a failed discovery's one report is its line,
    # not a toast too (+0): 33,942; Ctrl+Q asks about staged drafts before
    # the theme prompt, whose Discard undoes a Try (+0): 33,942.
    # Capture fix 3: Save selected's line names the provider and counts in
    # words, not "model(s) to <config key>" (+3): 33,945.
    # Capture fix 3, group C (ruling C-N1): the snapshot Settings leaves
    # names its unsaved categories, so a quit from another screen asks;
    # the stand-in lives in settings_quit_guard.py (+4): 33,949 with both.
    # Capture fix 3 round 1: save_state names them after every capture, the
    # raw editor's queued input included (+1): 33,950; and name only what
    # it keeps, not the snapshot preferences leaving drops (+1): 33,951;
    # Save selected reports once, its line or (after Discover or Clear) a
    # toast in the line's words (-1): 33,950.
    # Capture fix 3 round 2: a Save selected that saves nothing after
    # Discover or Clear says why in a warning toast, and a raised one is
    # logged first (+4): 33,954.
    # Capture fix 3 round 3: that toast names the provider the save was for
    # (a provider change takes the line too), and a raised save's next step
    # is one the dropped rows leave possible (+12): 33,966.
    # Capture fix 3 round 4: that step follows what is on screen (a failed
    # Clear keeps the checked rows; a provider change says to switch back)
    # (+13): 33,979.
    # Capture fix 3 round 5: a Clear still running when the save fails
    # counts as dropping the rows, so the step is never "Try Save again"
    # (+7): 33,986.
    # Rebased onto dev (2026-10-13): dev grew the file from 34,606 to 36,393 lines
    # since this PR's base (the Guardian rule editor and Personal Context
    # owners, among others), and this PR's net change is the same as before:
    # 35,771.
    # Round 4 after the rebase (ruling cf3r1-N1): leaving Settings keeps
    # unsaved snapshot preferences; the keep and restore live in
    # settings_quit_guard.py (+4): 35,775.
    # Post-rebase round 1: leaving Settings waits for a snapshot Save still
    # writing; the wait lives in settings_quit_guard.py (+3): 35,778.
    # Capture 13c: save_state starts from a fresh dict, so a kept snapshot
    # edit is forgotten once saved or reverted (+1): 35,779.
    # The Speech draft is kept and restored with replace(), so its pipeline
    # voice pair is no longer dropped (-13): 35,766.
    # Undoing an audio.cpp review merge copies the draft with replace() and
    # compares it on every field, so a pipeline voice edit survives (+1): 35,767.
    # Rebased onto dev 2026-10-15: dev's UI-perf waves (tasks 34913, 34934,
    # 34935-34937 and their review fixes) added 51 lines; this PR's own net
    # change is unchanged: 35,818.
    # Phase 9 Task 2, TASK-33002.9: an unlisted model leads the Test rows and
    # its toast warns (+3), offset by two reflowed statements (-8): 35,813.
    # TASK-33005.7: the Key block's identity row moves to
    # Settings_Modules/readiness_lines.py and gains its word (-4): 35,809.
    # TASK-33005.15: the Overview's Configuration row moves there too, and
    # its readiness poll keys on the evidence revision (-28): 35,781.
    # Its review round 1: the card's Key rows refresh writes the identity row
    # from the Key check row's verdict, so its own write here goes (-3): 35,778.
    # Phase 9 Task 7 (TASK-33002.10, .11, 33005.8, 33007.11): its lines are
    # offset by dropping three evidence lookups' None guards (evidence_for
    # refuses None itself) and the stale check's sentinel pair (net -1): 35,777.
    # Phase 9 Task 12, TASK-33005.10: the Test rows' model facet is the
    # snapshot's own model_facet, not a second copy (-5): 35,772.
    # TASK-33005.15's PR review: one write covers every Overview row (-1): 35,771.
    # Rebased onto dev 69771c1811, whose TASK-33008 keyless-403 hint is -1: 35,770.
    # Rebased onto dev 88fe3108e5, whose task-35150 Tools & MCP category is +1: 35,771.
    # TASK-35010: the saved lists are re-read from the file, not appended to in
    # memory, so the 20-line in-memory append goes (-4 net): 35,766.
    # Its review: Config reload's file re-read moves to the worker (-1): 35,765.
    # TASK-35011: Save selected's receipts, retry steps and empty-Model fill
    # move to Settings_Modules/saved_model_list.py with the new pick-joins-
    # on-s logic, and the duplicate capability line goes (-19): 35,746.
    # Rebased onto dev 1bbc145ffc, where #3169 merged at 35,771 (+1 over
    # S1's base of 35,770): 35,747.
    # TASK-35677/35678: bounded paste ownership and action support (-13).
    # TASK-35680: parallel-run origin handling removes one more line.
    # TASK-36098 retains dev decomposition and moves media completion out.
    # 2026-10-22, TASK-34000.27, owner decision ("expand the limits, don't modify
    # other code to make it fit", 2026-10-04): dev 7700c63a68's 35662 -> 35668
    # (+6) -- the audio_cpp fence's QueryError arm says why it refused a nav
    # switch instead of rolling back silently.
    # TASK-35647: the Tools & MCP badge branch (+2) folds two identical badge
    # branches (-2). TASK-35712: s names a model id the save refuses (+14);
    # two methods nothing calls go (-19): 35,663.
    # TASK-36361: staged completion delegates to the same-owner helper (-45).
    # TASK-36387: five badge branches read the category list's table (-3).
    # TASK-36388: the context-window suppress queue nothing wrote goes (-13).
    "tldw_chatbook/UI/Screens/settings_screen.py": 35602,
    # TASK-33007.8: the two Settings region modules Phase 7 created, pinned
    # at birth so the split cannot regrow a god module there. New code for
    # either goes in a sibling module under UI/Settings_Modules/.
    # providers_models_card.py: the final review's ruff format (+2 reflow)
    # and fixes re-pin it at the PR head's measured 1,963; the full-screen
    # capture fixes (2026-10-09: the restored-connection row, the Applies-to
    # line, Sign in with's row copy) at 2,194; the Endpoint guide that says
    # required as its row does (2026-10-09) at 2,238; items 11-12 (one-row
    # Context window, -1; its assumed size, +1) leave it at 2,238 measured.
    # dev's markup-interpolation gate (TASK-1513) found two sites here, and
    # the test written for the first found a third: a model id holding "[/]"
    # raised MarkupError from the Custom ID roll-back toast and from the Key
    # block's readiness row (which crashed the card at mount, on dev too);
    # one tooltip interpolated bare. Three guards, +3. The review of #3057
    # found two more rows that print an endpoint or an error's text through
    # markup (the save result, the discovery status): +2. That review's
    # fixes (2026-10-11) re-pin it at the integrated head's 2,371: the API
    # key row follows an unsaved Env var edit and its sentences fit one line
    # (R1 #10, #11), with the Source words as named constants (C3, C4). Fix
    # round 1 lowers it to 2,367 (the custom-endpoint sentence is the
    # screen's own; the quit summary's controls read the screen's values).
    # Fix round 2: an Env var edit drops the "invalid" verdict only when the
    # saved name is what is invalid, read only then (+5): 2,372. The capture
    # fix wave (2026-10-12): the Inspector's unsaved-edits note is one row
    # and says why (+3): 2,375; the Saved model list's duplicate
    # empty-state line is gone (-8): 2,367; Clear shows only while there
    # is a key to clear, one rule for the card and the screen, and a
    # keyless provider's Env var reads optional (+39): 2,406. Capture fix
    # round 1: the API key help's width comment says what is measured (+0).
    # Capture round 2 A1: a filled Endpoint's help drops "required:" (+1):
    # 2,407; the Saved model list names the saved models it does not show,
    # and the discovery line hides when a find leaves it empty (+43): 2,450;
    # the unsaved-edits note names the unsaved fields (+17): 2,467.
    # Capture fix A2 round 2 (2026-10-13): its checkboxes say their state in
    # the glyph, a check or a blank (StateCheckbox, +1): 2,407.
    # Integrated capture fix 2 (2026-10-13): re-measured after the pick: 2,468.
    # TASK-35011: the Saved model list block and its title summaries move to
    # saved_model_list.py (-100): 2,365.
    "tldw_chatbook/UI/Settings_Modules/providers_models_card.py": 2365,
    # settings_field_rows.py: the review of #3057 lowers it from 729 to 722
    # (two constants it had copied are imported; one number spelling, C6).
    # The capture fix wave (2026-10-12): Temperature's placeholder and
    # Validation line state the provider's own range (+38): 760. Capture fix
    # round 1: UNSET_FALLBACK_HELP's comment says 44 cells, not 45 (+0).
    # Capture fix A2 round 2 (2026-10-13): Permission summaries' Provider
    # list, holding a saved spelling or naming one it lacks (+34): 794.
    "tldw_chatbook/UI/Settings_Modules/settings_field_rows.py": 794,
    # TASK-35011 (S1 T2): the Saved model list block, its receipts and the
    # discovered pick that joins the list on s, pinned at birth (the
    # TASK-33007.8 precedent). No PR lane runs this row's node: NO-2 (b),
    # PR #3036 touches derived-artifacts.yml. Its review round 1: a pick the
    # combined save would refuse (a sample id, an unwritable saved id) saves
    # the default alone, and the lists refresh whenever a pick was pending:
    # 505. Round 2: a key the save would refuse is named in its receipt: 516.
    # Round 3: a model id it cannot write is named, not the key: 518.
    "tldw_chatbook/UI/Settings_Modules/saved_model_list.py": 518,
    # TASK-33621.51 (2026-10-29, owner rule 2026-10-04: a fix over a size
    # ratchet raises the row by its own lines, dated, never squeezes other
    # code): +27 for the Automatic Library outcome row under a USER turn
    # (setter, row plan, Static build, protected click class, and the row
    # kind Literal naming library-outcome and library-activity). Raised by this
    # branch's delta only: dev ed7481a30c already measures 8,821 against the
    # old 8,353, and that overrun is not this task's to absorb, so the row
    # stays red until it is decomposed or re-pinned by its owner.
    "tldw_chatbook/Widgets/Console/console_transcript.py": 8380,
    # TASK-33003 (Phase 3) ends at 7,764 measured, down from 7,802: .2 moved
    # the control-height rules to app CSS, .4/.5/.8 spent part of that.
    # TASK-33006.1 lowers it to 7,516: the Model view's field rows, Source
    # words and open focus moved to console_settings_field_row.py.
    # TASK-33006.2 lowers it to 7,466: the support sync moved there too, and
    # the modal's own support table and "no effect" copy are gone. Its
    # review fix lowers it to 7,445: the required check and the control
    # support reads moved there as well. TASK-33006.3 lowers it to 7,410:
    # the Request estimate and name disclosures are built there now.
    # TASK-33006.4 lowers it to 6,201: the provider picker, the model search
    # and their adapters, Custom model and Keep unverified are gone; the
    # model changes only through Switch model's pick mode. Its review fix
    # lowers it to 6,189: a created entry's landing moved to the field rows.
    # TASK-33006.5 lowers it to 6,156: the footer buttons lost their
    # compose-time geometry and Use saved defaults lives in
    # console_settings_saved_defaults.py. TASK-33006.6 lowers it to 6,143:
    # the label column's inline widths moved to app CSS, so the Context view
    # can widen it. TASK-33006.7 lowers it to 6,142: one handler switches
    # both views, which now open at their top. The Phase 6 final review fix
    # lowers it to 6,103: Save as model default shows on the saved-defaults
    # answer, so the persisted-table predicate is gone. (Counts above predate
    # PR #2993's ruff reflow of this file, +60 lines AST-identical; Phase 6
    # rebased onto it measures 6,193.) TASK-33003.11 lowers it to 6,178: one
    # loop sets both view tabs' label, class and tooltip. The loop alone
    # measures 6,169; the other nine lines are the same plan task's next two
    # riders, the single mount-time readiness build (TASK-33006.21) and the
    # Context view's unsaved-endpoint reason (TASK-33006.22). The plan task's
    # review fix lowers it to 6,177: the subscription poll's second
    # default-readiness sync is gone (TASK-33006.21). Then to 6,155: compose
    # yields the two view tabs unmarked and leaves the mark to that loop
    # (TASK-33003.11). Then 6,160: that loop's tooltip escapes the tab name,
    # as the markup-interpolation guard requires (escape_markup import).
    # Phase 9 Task 6 lowers it to 6,159: a listing is kept, with the endpoint
    # it listed, through the field rows' helper (TASK-33006.19).
    # TASK-33002.7 shrank it to 6,153 (2629dafe36, then 4f1110488e).
    # Rebased onto dev 69771c1811, whose TASK-33008 keyless-403 hint added 13: 6,166.
    # TASK-35578: 6,159; the dead _snapshot_model went, the saved-list note came.
    # Its review: 6,162; a named custom endpoint's listing gets its own note.
    "tldw_chatbook/Widgets/Console/console_settings_modal.py": 6162,
    "tldw_chatbook/UI/MCP_Modules/mcp_workbench.py": 6760,
    # Tier-2 review S03/S04: the two largest TTS modules had no row at all,
    # though both are larger in CLASS terms than every row above them
    # (TTSProfileRepository 4,880 lines / 118 methods; TTSService 2,825 / 93)
    # and profile_repository.py is within 456 file lines of the smallest
    # governed row. Pinned at their exact measurement, like every row here.
    "tldw_chatbook/TTS/profile_repository.py": 6304,
    "tldw_chatbook/TTS/TTS_Generation.py": 4046,
    # Added by the tier-2 review (S06 P2 [D3]): the largest module in the
    # repo with no row -- 1,217 methods on one class, rank 8 repo-wide,
    # while rank 9 (`personas_screen.py`) was already pinned. The
    # hand-picked list above missed it. 1,086 of those methods are a single
    # `_request(...)` -> `Model.model_validate` shape, so the honest
    # decomposition is per-API-namespace delegates -- `MCPUnifiedClient` is
    # the precedent already in the package.
    "tldw_chatbook/tldw_api/client.py": 16687,
    # Added 2026-09-21 (task-32901), each at its exact measured size as of
    # `origin/dev` 9e33252708 — see the module docstring's second pass.
    "tldw_chatbook/UI/Screens/watchlists_collections_screen.py": 14283,
    # TASK-33921: 10,854 on dev 2026-10-03 (over by 450); the Voice step moved
    # to UI/Wizards/first_run_voice_step.py. TASK-34100.1: every other step
    # moved to its own module (9,866 -> 3,149), then the busy line and the
    # worker helper's call sites landed (3,183); review round 1 moved the
    # fence's focus hold into first_run_step_guard.py (3,182); round 2 held the
    # fence through finishing (3,181); round 3 shared the discovery reuse rule
    # (3,179). The wizard keeps the container, progress, nav bar, dialog and
    # screen. TASK-34100.8: the Voice resume restore moved onto the step
    # (VoiceSetupStep.restore_checkpoint) -> 3,145; review round 1 moved the
    # refused-Next suffix onto SetupStep.refusal_message -> 3,144.
    # TASK-33008 (Phase 8) -> 3,136.
    "tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py": 3136,
    # TASK-34100.1: the step modules the split left at 1,000 lines or more
    # are pinned at their exact size, so the split cannot regrow a god module
    # a step at a time: Provider, Speech, Voice (moved out by TASK-33921) and
    # Model. New code for one of them goes in a helper module beside the step
    # (first_run_<step>_<topic>.py), never in a raised row. The smaller step
    # modules and helpers (Welcome, RAG, Tools, Notes, Appearance, Protect,
    # Summary, the shared widgets, the discovery helpers, the busy line and
    # the step guard) have no row: this file pins hand-picked god modules, and
    # a row on a small module would stop the next fix adding even one line.
    # Provider 2,461 -> 2,458: the Qodo round moved the handoff clear into
    # first_run_model_discovery.drop_unreusable_handoff. TASK-33008 review:
    # the Test-evidence line moved there too -> 2,456. Review round 2: the
    # failed-listing line and the key line went there as well -> 2,453.
    # Round 3: the key mark came back to the listing's call site, where a
    # reader sees it (that line stays pure) -> 2,454.
    "tldw_chatbook/UI/Wizards/first_run_provider_step.py": 2454,
    # owner decision 2026-10-04 (expand limits): TASK-34100.14 +60 -- the
    # download's cancel switch, its live status line and the cancelled
    # outcome are wired on the step (its @on handlers must stay on the
    # class); the copy itself lives in first_run_speech_step_state.py.
    # owner decision 2026-10-04 (expand limits): TASK-34100.14 +2 (review
    # F4) -- leaving setup abandons a Speech download still waiting in line;
    # then -4 (review F3), the use-from-disk VAD status copy moved to
    # first_run_speech_step_state.external_vad_install_status -> 1,766;
    # then +2 (review F5), a progress tick handled after the install's
    # result no longer rewrites the status line -> 1,768.
    "tldw_chatbook/UI/Wizards/first_run_speech_step.py": 1768,
    # TASK-34100.8: Voice 1,070 -> 1,014 -- its OmniVoice half moved to
    # first_run_voice_omnivoice.py and its Advanced fields to
    # first_run_voice_pickers.compose_voice_advanced while the step gained
    # the SF3 fixes, whose logic lives in first_run_voice_{prefill,status}.py.
    # Review round 1 -> 960: the OpenAI key lookup moved to
    # first_run_voice_credentials.py, and sample playback and the save-result
    # plumbing (shared by both halves) to OmniVoiceStepBase; the focus
    # hold/restore moved there too and the hint line became a property
    # -> 920. Review round 2 -> 892: the disabled-Test reason and the
    # replaced/kept copy moved to first_run_voice_{status,prefill}.py, and
    # the service table to first_run_voice_service_row.py.
    "tldw_chatbook/UI/Wizards/first_run_voice_step.py": 892,
    # TASK-33008 (Phase 8): 1,012 -> 994; its review's fixes -> 997.
    # TASK-35578: 990; the subtitle moved to first_run_model_discovery.
    "tldw_chatbook/UI/Wizards/first_run_model_step.py": 990,
    "tldw_chatbook/UI/Screens/llm_screen.py": 5180,
    "tldw_chatbook/UI/Screens/change_review_screen.py": 4967,
    # TASK-36942.1 (architecture review 2026-10-25, F16): every module of
    # 4,000+ lines that had no row in any of the three size ratchets, each
    # pinned at its measured size on dev 2b296e60b3. The review filed 40
    # modules at dev 7a5e2513c5; the count held at 40 and five sizes drifted
    # upward (ChaChaNotes_DB 25,565 -> 25,693, Console_Modules/workspace
    # 8,522 -> 8,747, console_composer_bar 6,958 -> 7,024,
    # unified_control_plane_service 5,917 -> 5,920, library_notes_canvas
    # 4,463 -> 4,646). Later phases lower these rows;
    # `test_every_module_of_the_guard_size_has_a_row` keeps new gaps from opening.
    # TASK-37270: removed the dead local map shadowed by TOML-derived defaults.
    # 2026-10-29, TASK-33622.6 (owner decision: a fix over a size ratchet
    # raises the row with a dated comment; never squeeze code to fit):
    # 25693 -> 25705 (+12). The conversation search gains `title_only` (the
    # Console rail filter's count and page match titles, as a literal
    # substring), threaded through the shared filter and the page query.
    # 2026-10-30: re-measured after rebasing onto dev 134108dd84 (owner rule: the row is the exact measurement): 25693 -> 25705.
    # 2026-10-27 (TASK-33620.10, owner decision: a fix over a size ratchet
    # raises the row): 12258 -> 12298 (+40) -- the shipped
    # `[chat_defaults] streaming = true` with its comment, and the one merge
    # helper every published config view now uses so that key cannot shadow
    # a file's legacy `enable_streaming`.
    "tldw_chatbook/DB/ChaChaNotes_DB.py": 25705,
    "tldw_chatbook/config.py": 12251,
    "tldw_chatbook/Chat/console_agent_bridge.py": 12088,
    "tldw_chatbook/Notes/file_notes_git_service.py": 11400,
    "tldw_chatbook/Agents/agent_service.py": 9620,
    "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py": 9532,
    "tldw_chatbook/DB/Client_Media_DB_v2.py": 9426,
    # TASK-37270: bounded template reads moved to local_reasoning_metadata.
    # 2026-10-29, TASK-33622.6 (owner decision: raise with a dated comment):
    # 8747 -> 8763 (+16). The flat Conversations lane filters its rows and
    # asks storage for its count by title only (owner decision 2026-10-27),
    # so the two can no longer disagree.
    # 2026-10-30: re-measured after rebasing onto dev 134108dd84 (owner rule: the row is the exact measurement): 8747 -> 8763.
    # 2026-10-29 (TASK-37421, owner decision: every self-hosted and Custom
    # OpenAI-compatible server gets llama.cpp's one-time non-streaming retry;
    # a fix over a size ratchet raises the row): 8184 -> 8333 (+149) -- the
    # generic path's retry in stream_chat, sharing the llama.cpp fallback's
    # two trace steps (hoisted, not duplicated), the refused/empty-stream
    # hand-back in _stream_generic_chat, and the retry's usage hand-over.
    # Classification and copy live in Chat/refused_stream.py.
    "tldw_chatbook/UI/Console_Modules/workspace.py": 8763,
    "tldw_chatbook/Chat/console_provider_gateway.py": 8179,
    # 2026-10-27, TASK-33620.15.22 (owner decision: a fix over a size ratchet
    # raises the row with a dated comment; never squeeze code to fit):
    # 7024 -> 7043 (+19). Send's readiness gets one owner,
    # ConsoleComposerBar.has_sendable_payload, read by both sync sites and
    # the Workbench, fed by the screen's send-gate attachment predicate.
    "tldw_chatbook/Widgets/Console/console_composer_bar.py": 7043,
    # TASK-37620: document the live workspace-alignment seam rather than
    # historical extraction call counts, retaining activation notifications.
    "tldw_chatbook/UI/Console_Modules/session.py": 6990,
    "tldw_chatbook/DB/Subscriptions_DB.py": 6987,
    "tldw_chatbook/Media/local_media_reading_service.py": 6911,
    "tldw_chatbook/Chat/console_runtime.py": 6686,
    "tldw_chatbook/Chat/console_trace_service.py": 6674,
    "tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py": 6108,
    "tldw_chatbook/Notes/notes_sync_executor.py": 6007,
    "tldw_chatbook/MCP/unified_control_plane_service.py": 5920,
    "tldw_chatbook/Agents/local_tool_provider.py": 5749,
    "tldw_chatbook/Tools/remote_worker_bundle.py": 5706,
    "tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py": 5532,
    "tldw_chatbook/LLM_Calls/LLM_API_Calls.py": 5186,
    "tldw_chatbook/Notes/notes_sync_runtime.py": 5167,
    "tldw_chatbook/Personal_Context/repository.py": 4997,
    "tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py": 4968,
    # owner decision 2026-10-04 (expand limits): TASK-34100.14 +43 -- the
    # cancelled download's stage discard (_discard_download_stage_for, run
    # while the session lease is still held) and its provision-continuation
    # allowlist entry; dev pinned this row while the branch was in flight.
    "tldw_chatbook/Model_Artifacts/service.py": 4896,
    "tldw_chatbook/Character_Chat/Character_Chat_Lib.py": 4775,
    "tldw_chatbook/DB/Prompts_DB.py": 4760,
    "tldw_chatbook/Chat/citation_trace_repository.py": 4719,
    "tldw_chatbook/Widgets/Library/library_notes_canvas.py": 4646,
    "tldw_chatbook/Web_Scraping/WebSearch_APIs.py": 4611,
    "tldw_chatbook/RAG_Search/simplified/rag_service.py": 4507,
    "tldw_chatbook/Local_Ingestion/transcription_service.py": 4494,
    "tldw_chatbook/Chat/chat_persistence_service.py": 4263,
    "tldw_chatbook/UI/Screens/evals_screen.py": 4211,
    "tldw_chatbook/Workspaces/registry_service.py": 4232,
    "tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py": 4207,
    "tldw_chatbook/UI/MCP_Modules/mcp_inspector.py": 4174,
    "tldw_chatbook/Dreams/track_service.py": 4061,
    "tldw_chatbook/DB/private_sqlite.py": 4030,
}

#: Same tolerance as the Library controller ratchet: loose enough that
#: ordinary in-file edits do not fail CI, tight enough that a real shrink
#: which forgot to lower its row is still caught.
_SLACK_TOLERANCE_LINES = 50


@lru_cache(maxsize=None)
def _measure(rel_path: str) -> int:
    """Line count of a module via ``str.splitlines()``.

    Cached because both tests below parametrize over the same paths.

    Raises:
        AssertionError: If the module is missing — the budget entry is
            stale and must be updated deliberately, not silently skipped.
    """
    path = _REPO_ROOT / rel_path
    assert path.exists(), f"{rel_path} not found; the budget entry is stale."
    return len(path.read_text(encoding="utf-8").splitlines())


@pytest.mark.unit
@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))
def test_module_does_not_grow_past_its_budget(rel_path: str) -> None:
    """The ceiling itself: a budgeted module may not exceed its pin."""
    max_lines = _BUDGETS[rel_path]
    lines = _measure(rel_path)

    assert lines <= max_lines, (
        f"{rel_path} grew to {lines} lines (budget {max_lines}, "
        f"+{lines - max_lines}).\n\n"
        f"{rel_path} is under a size ratchet "
        f"(Tests/Architecture/test_module_size_ratchet.py). Put new code in "
        f"a controller/widget/helper module — do NOT raise the budget to "
        f"make this pass. Lower it when the file shrinks."
    )


@pytest.mark.unit
@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))
def test_budget_is_not_left_slack(rel_path: str) -> None:
    """The recorded budget should track reality, not drift above it."""
    max_lines = _BUDGETS[rel_path]
    lines = _measure(rel_path)

    assert max_lines - lines <= _SLACK_TOLERANCE_LINES, (
        f"{rel_path} is {max_lines - lines} lines under its budget "
        f"({lines} vs {max_lines}). Set it to {lines} so the real "
        f"measurement is what's pinned."
    )


#: A module this long needs a row in one of the three size ratchets.
_ROW_REQUIRED_AT_LINES = 4000


@pytest.mark.unit
def test_every_module_of_the_guard_size_has_a_row() -> None:
    """No tldw_chatbook module reaches 4,000 lines without a budget row."""
    rowed = set(_BUDGETS) | set(_SCREEN_BUDGETS) | set(_LIBRARY_BUDGETS)
    package = _REPO_ROOT / "tldw_chatbook"
    unbudgeted = sorted(
        rel
        for path in package.rglob("*.py")
        if (rel := path.relative_to(_REPO_ROOT).as_posix()) not in rowed
        and len(path.read_text(encoding="utf-8").splitlines()) >= _ROW_REQUIRED_AT_LINES
    )
    assert not unbudgeted, (
        f"Module(s) of {_ROW_REQUIRED_AT_LINES}+ lines with no size ratchet "
        f"row: {unbudgeted}. Add a row to _BUDGETS in "
        f"Tests/Architecture/test_module_size_ratchet.py pinned at the "
        f"measured size, or split the module below the line."
    )
