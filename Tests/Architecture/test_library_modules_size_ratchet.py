"""A size ratchet for `Library_Modules` controller files — self-defending.

**Why this test exists.** task-31203 AC#4 (Library decomposition wave 3,
the combined search+RAG series) requires governance for the controller
files landing under `tldw_chatbook/UI/Library_Modules/` as each
subsystem's extraction series completes — the same problem
`test_screen_size_ratchet.py` solved for `chat_screen.py`/
`library_screen.py`, but for the destination files a decomposition wave
writes INTO rather than the source it extracts FROM. As of this writing
five controllers exist (conversations reader/browse, export, collections
capture/main) at 300–2,000+ lines each, with wave 3 (search+RAG) and six
more subsystems still to land — nothing has governed their size until now,
even though the wave-2 final review flagged exactly this gap
(`library_collections_controller.py` at 1,689 lines, `library_
conversations_controller.py` at 1,738, called out by name).

**The design tension this test resolves.** The byte-for-byte canon
(`backlog/docs/library-decomposition-recipe.md` §1) moves method bodies
verbatim into their subsystem's controller — every sanctioned move commit
therefore INFLATES the receiving controller by design (`library_
collections_controller.py` grew by 64 method bodies in one PR; `library_
conversations_controller.py` by 40). A ratchet that only ever allows a
number to go DOWN, read literally, would make every wave's own
controller-PR fail its own governance the instant it lands — the exact
opposite of what `test_screen_size_ratchet.py` exists to catch (silent
creep between waves, not the wave's own sanctioned work). Governance here
must distinguish two things that both show up as "the file got bigger": a
sanctioned move (the number goes up because a cluster of real methods just
arrived, verified by the same PR that adds a wiring test for them) versus
ordinary feature-code creep landing in a controller between moves, because
— like `library_screen.py` before Task 1 of the Library decomposition plan
existed — the controller was the path of least resistance.

**The two automated checks, and what they do NOT do.** Exactly like
`test_screen_size_ratchet.py`, no test here can read intent, so this file
cannot mechanically tell "a sanctioned move" apart from "creep." What it
CAN do is what the screen ratchet already does in practice — see that
file's own wave-2 final-review precedent, where `_BUDGETS` rows were
RAISED twice, each time with a dated comment explaining why: a ceiling
that can only move in a diff a reviewer sees, with a reason attached, is a
fence a silent regression cannot climb even though a reviewed, justified
move can. The enforcement is therefore:

1. `test_controller_does_not_grow_past_its_budget` — a file may never
   exceed its pinned ceiling without that pin being edited in the same
   diff.
2. `test_budget_is_not_left_slack_after_a_move` — the pin may not sit far
   above the real measurement either, so a wave that raises the ceiling
   generously "to be safe" and then never lands, or overshoots its own
   landing, is caught too.

**Re-pin flow at a sanctioned move**: re-measure after the move lands
(`len(path.read_text(encoding="utf-8").splitlines())`, the same expression
`_measure` below uses) and set the row to the exact new value, in the SAME
commit — identical to the screen ratchet's own §6 rule
(`backlog/docs/library-decomposition-recipe.md` §6), same file, same
commit, never deferred to a follow-up.

**Why line count only, with no method-count column** (unlike the screen
ratchet, which tracks both). The screen ratchet's method count exists to
catch a class made shorter by *compressing* bodies rather than by
extracting responsibility — line count alone could not tell those apart on
`ChatScreen`/`LibraryScreen`, which are each exactly one class filling
their whole file, always named after the screen. Controller files under
`Library_Modules/` do not have that one-class shape: alongside the primary
controller/coordinator class, the byte-for-byte canon's
constructor-dependency-binding pattern (recipe §1) deliberately produces
small immutable helper classes in the SAME file — Protocol ports, request
"fences," result "receipts," and outcome snapshots (see
`CaptureRequestFence`/`CaptureArchiveReceipt` in `library_collections_
capture_controller.py`, or the `*Port` protocols in `library_notes_sync_
controller.py`). There is also no reliable filename convention for picking
"the" dominant class the way the screen ratchet does (`ChatScreen` in
`chat_screen.py`): `library_skill_import_controller.py`'s primary class is
`LibrarySkillImportCoordinator`, not `...Controller` — a naive
filename-derived class-name lookup would silently miss it entirely. Given
that, either (a) a per-file class-name override table would be needed
(reintroducing exactly the kind of hand-maintained list this test's
glob-based discovery was chosen to avoid), or (b) summing methods across
every class in the file would count the very helper-class proliferation
the canon encourages as if it were controller-responsibility growth,
punishing a pattern the recipe recommends. File line count has neither
problem, and it is the exact axis the wave's own design tension (moves
inflate lines) is stated in. A future row for a controller file that
happens to be genuinely one dominant class with no helper types could add
a method-count column for that row specifically without disturbing this
reasoning for the rest — none currently qualify.

**The self-defending property the screen ratchet does not have.**
`_BUDGETS` in `test_screen_size_ratchet.py` is a hand-maintained dict with
nothing that notices when a NEW screen file should have a row and does
not — that gap is exactly how `library_screen.py` went ungoverned for a
month while it tripled in size. This file instead globs
`tldw_chatbook/UI/Library_Modules/*_controller.py` at collection time and
asserts every match has a `_BUDGETS` row:
`test_every_controller_file_has_a_budget_row` fails, by name, the moment a
new wave-3+ controller module lands without one, with guidance on the row
to add. New controllers are therefore born governed: nothing needs to
remember to edit this test file when wave 3's search/RAG controller(s)
are created — only to add their row once the failure names them.

**What to do when this test fails.**
- Growth past the ceiling with no move in flight: this is the creep this
  test exists to catch. Reconsider whether the new code belongs in this
  controller at all — do NOT raise the number just to make the test pass.
- A sanctioned move just landed: re-measure and set the row to the exact
  new value, in the SAME commit, with a one-line dated comment (mirror the
  screen ratchet's comment trail immediately above its own `_BUDGETS`).
- An unlisted controller file: add its row at its current exact
  measurement — do not pick a round number "for headroom."
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY_MODULES_DIR = _REPO_ROOT / "tldw_chatbook/UI/Library_Modules"
_CONTROLLER_GLOB = "*_controller.py"

#: path -> max lines. LOWER when a cleanup PR shrinks a controller; RAISE
#: (with a dated comment naming the landing PR) when a sanctioned move PR
#: adds a cluster of methods. Never raise to silence ordinary creep.
#:
#: First recorded 2026-09-03 by task-31203 AC#4 (Library decomposition
#: wave 3, Task 1 — controller-file size governance), at each file's exact
#: measured line count as of the wave-2 close (conversations, export, and
#: collections series all landed; wave 3's own search+RAG controller(s)
#: not yet created, so not yet rows here — they are born governed by the
#: glob in `test_every_controller_file_has_a_budget_row` the moment they
#: exist, and get their own row at that point).
_BUDGETS: dict[str, int] = {
    # Re-pinned 2026-09-19 (core-review TASK-32809.1). The ratchet had
    # gone red on dev: 15 controller rows had been overrun by feature
    # work landing in the controllers between moves (the core-tests job
    # collects this file, so it was failing). Per the recipe's
    # pin==measurement rule (backlog/docs/library-decomposition-recipe.md
    # §6) each red row below is re-set to its exact current line count so
    # the ceiling tracks reality again and the file can only shrink from
    # here; nothing is raised above measured. This is a baseline reset,
    # not a sanctioned decomposition move -- decomposition candidates are
    # recorded separately (TASK-32809.3), none opened here.
    # 2026-09-05, wave-6 final review (`origin/dev` reconciliation merge):
    # BOTH rows below are dev-side controllers, not this wave's work. Dev
    # created `library_character_repair_controller.py` and `library_
    # navigation_controller.py` without adding either to this table, so
    # `test_every_controller_file_has_a_budget_row` -- the self-defending
    # property this file exists for -- named them the moment the merge
    # landed them here. That is the glob doing its job on someone else's
    # miss; they are governed at this merge rather than left to whichever
    # branch next trips the check. Pinned at their exact measured line
    # counts (`len(path.read_text(encoding="utf-8").splitlines())`, this
    # file's own `_measure` expression), no headroom.
    "tldw_chatbook/UI/Library_Modules/library_character_repair_controller.py": 518,
    "tldw_chatbook/UI/Library_Modules/library_collections_capture_controller.py": 699,
    "tldw_chatbook/UI/Library_Modules/library_collections_controller.py": 1689,
    # TASK-32659: bounded saved-search loading, including authority/request fencing.
    "tldw_chatbook/UI/Library_Modules/library_collections_saved_search_controller.py": 49,
    # TASK-33628.10: 976 -> 964. The ensure-selection decision moved to
    # library_conversation_reader_freshness.py with its new transcript re-check.
    # 964 -> 967 (2026-10-03, Qodo #2989): a completed load calls that module's
    # recheck_settled_load, so a load a list read found in flight is re-checked.
    "tldw_chatbook/UI/Library_Modules/library_conversation_reader_controller.py": 967,
    "tldw_chatbook/UI/Library_Modules/library_conversations_controller.py": 1800,
    # 2026-10-21, TASK-34000.22 fix round 1 (owner decision 2026-10-04: raise
    # the row to the measured value): 1453 -> 1492. Dev itself was already
    # over this row at 1484 (one of dev's own pre-existing ratchet reds);
    # this task adds 8 lines -- the Media authoring forms' dirty refusal
    # beside the Export… opener's note flush (C-1).
    "tldw_chatbook/UI/Library_Modules/library_export_controller.py": 1492,
    # 2026-09-05, wave-5 task 2 (ingest controller PR, series 2/3): born
    # governed the moment this file existed (task-31203 AC#4's glob-based
    # discovery, recipe §17) -- 57 moved methods (byte-for-byte) + a
    # 37-parameter constructor (self + screen + 35 named dependencies; see
    # the module's own docstring for the full 78-candidate/21-exclusion
    # derivation, including 6 instance-attribute-monkeypatch exclusions
    # found only by running the battery) pinned at its exact measured line
    # count.
    # Task 2 fix round 1 (post-review): `_resolve_ingest_source` excluded
    # (module-globals coupling on `validate_path_simple`/`validate_url`,
    # found by the mechanical module-globals census, not the battery --
    # see the screen ratchet's own comment above for the full incident);
    # 56 moved methods, one named dependency added for its mover caller.
    # Constructor arity verified with `inspect.signature(LibraryIngest
    # Controller.__init__)` rather than hand-counted: 37 -> 38 params excl.
    # `self` (1 positional `screen` + 36 -> 37 keyword-only named
    # dependencies), a clean +1 matching the one new dependency. Dead
    # imports removed (`logger`, `validate_path_simple`, `validate_url` --
    # all now unused in this file). 2510 -> 2536 lines.
    #
    # 2026-09-05, wave-5 task 3 (ingest cleanup, series 3/3): comment-only
    # growth, no method body touched (56 movers unchanged -- cleanup prunes
    # the SCREEN's delegators, not the controller's own methods, per the
    # collections/search+RAG/skills precedent). Four moved-docstring-
    # adjacent module/constructor docstring corrections: two stray "63"
    # counts (should be 56 -- an arithmetic slip inherited from an earlier
    # draft, never 78-22 or any other real derivation), the "LibraryScreen
    # keeps one-line delegators under every one of these 56 original names"
    # claim (now false for 6 of the 56, this task's own screen-side prune),
    # and the `_apply_library_ingest_backend_save`/`_sync_library_canvas`
    # module-globals census evidence (corrected from 7 files/~20 sites to
    # the true 10 files/38 sites -- 3 files missed by task 2's own grep
    # because their patch sites used a variable name other than
    # `library_screen`/`library_screen_module`; see the recipe's §3 for the
    # full correction) -- all fixed in this task, +22 lines. 2536 -> 2558.
    #
    # 2026-09-05, wave-5 task 3 fix round 1 (post-review, counts only):
    # comment-only growth, no method body touched. The `_sync_library_
    # canvas` census's own "site" definition (one match/line of the
    # 3-shape pattern set, deduplicated per file) and its reproducible
    # 10-file breakdown summing to 38 were added to this module's own
    # docstring so a reviewer can re-derive the number without re-running
    # the census script. 2558 -> 2569.
    #
    # 2026-09-05, wave-5 final review: `origin/dev` merge (89 commits since
    # this branch's merge-base 68f9d865f). Dev edited one ALREADY-MOVED body
    # after the move landed -- `handle_library_ingest_clear_finished` gained
    # a 13-line stale-outcome prune (task-28007's Qodo review round) -- so
    # the edit follows the body here rather than resurrecting the screen's
    # copy: +13 ported lines, +5 for the `library_ingest_analyze_outcomes_
    # accessor` constructor parameter and its 4-line comment, +3 for storing
    # it, +12 for the same-named property that lets the ported lines stay
    # byte-for-byte with dev's, and +21 of module docstring recording the
    # divergence, the new group-(b) binding, and the follow-up task
    # (task-31651) that folds the field into `LibraryIngestState` proper.
    # No other body touched; 56 movers unchanged. 2569 -> 2623.
    #
    # 2026-09-05, wave-5 ROUND-2 `origin/dev` merge (72 commits since the
    # previous reconciliation's merge-base 93388ba69). Dev edited TWO more
    # already-moved bodies, both with TASK-31521 suspend gates (the Library
    # route became reusable, so navigation suspends the screen instead of
    # unmounting it): `_handle_library_ingest_registry_changed` defers its
    # dynamic-region rebuild, footer re-registration, source-snapshot
    # re-read and landing-attention sync while hidden, and `_handle_library_
    # ingest_progress_changed` skips pure DOM patching. Same rule as last
    # round -- the edit follows the body, the screen keeps its one-line
    # delegators: +20 ported lines (exactly dev's own delta on those two
    # bodies, so both port byte-for-byte), +15 for three new constructor
    # parameters and their comments (`library_screen_suspended_accessor`,
    # `library_ingest_suspended_activity_accessor`, `set_library_ingest_
    # suspended_activity`), +7 for storing them, +32 for the two same-named
    # properties (one getter-only, one getter/setter) that keep the ported
    # lines byte-for-byte, and +24 of module docstring (group (b) 8 -> 10,
    # and the divergence paragraph rewritten from one body to three).
    # Keyword-only constructor arity measured, not assumed: 38 -> 41 (43
    # total including `self` and `screen`). No other body touched; 56 movers
    # unchanged. 2623 -> 2721.
    # 2026-10-23, TASK-34000.20 (owner decision 2026-10-04: a fix over a size
    # ratchet raises the row to the measured value). dev `d0d926e11b` was
    # ALREADY over this row before the P1-C batch (measured 3087 vs 3080).
    # This task adds `_library_ingest_preflight_is_current`, the wait-gate at
    # the top of `_submit_library_ingest_form` (Enter and Start now import
    # only on a pre-check for the current text), the armed-submit fire in
    # `_apply_library_ingest_preflight_result`, the debounce stop + arm
    # re-point in `_trigger_library_ingest_preflight`, and two disarm sites:
    # 3080 -> 3164 (measured).
    # 2026-10-23, TASK-34000.20 fix round 1 (same owner decision): the armed
    # submit is bound to its exact text and fires only while the ingest row
    # is selected, the Library screen is active and the field still shows
    # that text (`_arm_/_disarm_library_ingest_submit`,
    # `_library_ingest_submit_may_fire`, `_library_ingest_preflight_worker_
    # is_live`), a press on a current not-found verdict re-checks, and the
    # trigger normalises whitespace: 3164 -> 3242 (measured).
    # 2026-10-23, TASK-34000.20 fix round 2 (same owner decision): the fire-time
    # text check reads the live path Input instead of the form echo
    # (`_library_ingest_submit_may_fire`): 3242 -> 3247 (measured).
    # 2026-10-24, TASK-34000.20 P1-C final fix round (same owner decision): a
    # press on a CURRENT verdict that shut the gate for what the check found
    # (empty folder, nothing importable, nothing the server takes) re-checks
    # like the path problem does (`_library_ingest_gate_is_shut_by_the_
    # precheck`, I-B): 3247 -> 3281 (measured).
    "tldw_chatbook/UI/Library_Modules/library_ingest_controller.py": 3281,
    # Re-pinned to the measured value 2026-09-19 by the core-review ratchet
    # re-baseline (TASK-32809.1), which SUPERSEDES the prior Library-wave
    # stance of holding this row at 371. That stance was correct FOR A
    # LIBRARY EXTRACTION PR (raising a dev-owned creep pin there would
    # launder dev-side debt behind a Library merge). This is not that: the
    # core review's explicit, reviewed job was to make every red ratchet
    # green again at its true current size so the ceiling is meaningful and
    # the file can only shrink from here. The creep is still dev's and still
    # needs a decomposition owner (recorded in
    # backlog/docs/size-decomposition-candidates-2026-09-18.md); this pin
    # just stops the ratchet lying about where the file is.
    "tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py": 720,
    # 2026-09-06, wave-7 task 2 (media controller PR, media series 2/3): born
    # governed the moment this file existed (task-31203 AC#4's glob-based
    # discovery, recipe §17) -- 140 moved methods (byte-for-byte; every one
    # verified TEXT-identical AND AST-identical to its pre-move screen body,
    # in both directions) plus the constructor/property scaffolding the canon
    # requires, pinned at its exact measured line count. Constructor arity
    # MEASURED with `inspect.signature(LibraryMediaController.__init__)`,
    # never hand-counted: 85 parameters including `self` -- 1 positional
    # (`screen`) + 83 keyword-only named dependencies (1 state accessor + 11
    # read-only shared-shell-state accessors + 5 read/write pairs = 10 + 2
    # media-wiring-controller accessors + 24 shell helpers + 35 late-binding
    # callables for the exclusions). 174 class-level `property` objects: 92
    # hand-written bindings + the 82 generated flat-name state shims. See the
    # module's own docstring for the full 251-candidate / 111-exclusion
    # derivation and the single-controller decision (one connected component
    # of 213 by call graph, 238 once shared-state edges are added, out of
    # 251 -- no second component of any size, and the plan's own candidate
    # browse-vs-trash seam measured at 16 cross-call edges and 6 shared state
    # fields).
    #
    # 2026-09-06, wave-7 task 2 fix round 2 (post-review, docstring only):
    # comment growth, no method body touched (140 movers unchanged, all still
    # byte-for-byte, re-verified TEXT- and AST-identical after the edit).
    # Three census claims in this module's own docstring were corrected in
    # place: exclusion class 2 asserted that all 16 instance-monkeypatch names
    # have a MOVER calling them, which a re-derived caller map shows is true
    # of 9 and false of 7 (those 7 are held by recipe §3's OPENING rule --
    # a patched name keeps its whole call graph screen-routed until cleanup --
    # not by a demonstrated bypass, so the exclusions stand and only the
    # stated reason changes); the generalized bare-`self` census returns 7
    # over the FINAL 140 movers, not the 10 it returned at the moment of the
    # finding when the three `ancestors` methods were still movers (both
    # figures now stated, because only the pair shows the census working);
    # and the `self in <widget>.ancestors` shape has FOUR sites in the
    # 251-candidate cluster, not three -- `_library_media_settlement_tree` is
    # the fourth and is already excluded under class 2, which is exactly how
    # a census that stops at newly-excluded methods undercounts the shape.
    # 4461 -> 4496.
    #
    # 2026-09-07, wave-7 final review: `origin/dev` reconciliation merge (306
    # commits, merge-base 761416317). Re-measured on the MERGED tree with this
    # file's own `_measure`: 4496 -> 4612 (+116), itemised so the number is
    # not an estimate:
    #   +63  the ELEVEN mover bodies dev edited on its side of the range. The
    #        screen keeps its delegator and dev's edit follows the body here
    #        (the wave-5 precedent). Each port is AST- AND text-IDENTICAL to
    #        dev's own `LibraryScreen` version, which is possible only because
    #        all 11 controller bodies were still byte-identical to the
    #        merge-base screen bodies (verified before porting) and because
    #        the shims below expose the same names.
    #   +8   eight new group-(e) keyword-only parameters,
    #   +8   their constructor assignments,
    #   +37  their late-binding properties plus the block's new comment.
    #        The eight are dev's NEW screen methods that the ported bodies
    #        call (`_after_library_media_viewer_sync`,
    #        `_decorate_library_media_reviewed`,
    #        `_library_media_can_rename_speakers`,
    #        `_library_media_list_unselectable`,
    #        `_library_media_selected_backing_id`,
    #        `_library_media_trash_action_disabled_reason`,
    #        `_queue_after_library_media_viewer_recompose`,
    #        `_sync_library_media_surfaces_or_recompose`). They stay
    #        screen-resident; keyword-only arity 83 -> 91.
    # No other body was touched: the 140 movers are otherwise unchanged.
    #
    # 2026-09-07, wave-7 ROUND 2 (dev's 75-commit `fix/media-riders` race):
    # 4612 -> 4630 (+18), itemised:
    #   +7   the TWO mover bodies dev edited
    #        (`_select_library_media_reader_row` +5,
    #        `handle_library_media_review_selected` +2). Both ports verified
    #        TEXT- and AST-identical to dev's own screen version, on the same
    #        precondition as round 1 (both controller bodies were still
    #        byte-identical to dev's pre-edit version before porting).
    #   +6   a NINTH group-(e) binding
    #        (`_restore_library_media_reader_width_on_open`, dev's new screen
    #        method that the ported selection body now calls): 1 signature
    #        parameter, 1 assignment, 4 property lines. Keyword-only arity
    #        91 -> 92.
    #   +5   an IMPORT, not a binding: the ported
    #        `handle_library_media_review_selected` reads
    #        `library_media_int_backing_id` as a BARE MODULE GLOBAL (dev
    #        promoted it out of a method), so it resolves against THIS
    #        module's `__globals__` -- recipe SS3's module-globals-coupling
    #        shape, which no `self.<attr>` census can see. Imported here with
    #        a 4-line comment saying so.
    #
    # 2026-09-08, wave-8 close (BEHAVIOUR FIX, the wave's one deliberate
    # behaviour change): 4630 -> 4669 (+39), and every one of those lines is
    # ONE new binding property plus its docstring -- `_media_state`, this
    # cluster's own state object under the screen's own name, the exact shape
    # `library_notes_controller.py` already carries. No moved body was touched
    # (all 140 movers unchanged, still byte-for-byte) and no moved body spells
    # `self._media_state`: they read the flat `_library_media_<field>`
    # properties from the generated shim loop. The accessor exists for the
    # SHARED `canvas_sync.py` dispatchers, whose media leg assigns through
    # `screen._media_state.selected_media_id` while
    # `handle_library_media_select_all`/`_select_clear` hand them a bare
    # controller `self` -- so before this, every media "Select all"/"Clear"
    # press raised `AttributeError` into the dispatcher's own
    # `except Exception` and silently took the whole-screen recompose.
    # Mutation-verified BOTH ways: removing this property reds
    # `test_media_row_toggle_resolves_the_dotted_state_path[controller]` and
    # only that leg; reverting `canvas_sync.py`'s media branch reds only the
    # `[screen]` leg.
    # 2026-09-09, Library phase C task 3 (region ownership, media): 4669 ->
    # 4670. Exactly one line, and it is the wiring the migration exists to
    # make possible -- `actions=self` at this module's single
    # `LibraryMediaCanvas(...)` construction site, which is what lets the
    # canvas own its sixteen canvas-origin `@on` rows without reaching back
    # through the screen. The screen shrank 63 lines in the same commit.
    # 2026-10-06, perf wave 5 task 12 (media filter debounce): 4768 -> 4769.
    # Exactly one line: the `MEDIA_FILTER_DEBOUNCE_SECONDS` import that the
    # filter handler now arms its timer with (it previously reused
    # SELECTION_SETTLE_SECONDS, a row-selection paint-settle value, as its
    # search debounce). No moved body was touched.
    # 2026-10-19, TASK-34000.24 (owner decision 2026-10-04: a fix over a size
    # ratchet raises the row to the measured value): 4769 -> 4874. origin/dev
    # 4de437a5c5 was ALREADY at 4812 against this row (one of dev's own
    # pre-existing ratchet reds, 43 over); this task added 62 lines: the
    # "Use in Console" link receipt threaded through the Reader build and
    # the unchanged-compare (+10), the receipt's visibility rule
    # `_library_media_visible_link_receipt` (+41), and the per-item clear at
    # the selection entry (+6, with its comment). No code was moved or
    # reflowed to fit. Re-measured after fix round 1: still 4874.
    # 2026-10-20, fix round 2 (item 4): 4874 -> 4898 (+24),
    # `library_media_link_receipt_block` -- why a standing receipt is
    # withheld (other item / registry unavailable / workspace inactive), so
    # the Undo handler can tell an unreadable registry from a moved
    # workspace; the visibility rule now reads it.
    # 2026-10-20, TASK-34000.18 (owner decision 2026-10-04: a fix over a size
    # ratchet raises the row to the measured value): 4898 -> 4933 (+35).
    # origin/dev 4de437a5c5 was ALREADY at 4812 against the old 4769 row
    # (one of dev's own pre-existing ratchet reds); the clarify batch
    # measured 4898 on 32caa91c1a. This task added `_library_media_reader_
    # target_id` -- the one identity every Reader-scoped action reads
    # (+14, docstring included), the trash confirm's `delete_target_title`
    # threaded through the Reader build and the unchanged-compare (+5) and
    # its resolver `_library_media_delete_target_title` (+13), plus the
    # import and three one-line reads. No code was moved or reflowed to fit.
    # 2026-10-20, TASK-34000.18 fix round 1 (same owner decision): 4933 ->
    # 4989 (+56). `_adopt_library_media_highlight` moved onto the controller
    # (+32 with its docstring; the screen keeps a one-line delegator) so
    # `_build_library_media_active_child` routes through it too, and the
    # one predicate `_library_media_reader_holds_highlight` (+22) decides
    # whether the Reader holds the highlight (viewer view AND unfiltered
    # scope), read by the builder input (+2) and the adopter.
    # 2026-10-20, TASK-34000.18 fix round 2 (same owner decision): 4989 ->
    # 5006 (+17). The hold is ORDER-based now: the predicate reads the
    # `highlight_held_by_reader` state field instead of the scope (docstring
    # +4), the row-select seam sets it (+3), and the list canvas's own
    # presentation resolves the list selection itself (+10) rather than the
    # Reader's target.
    # 2026-10-20, TASK-34000.22 (same owner decision; dev was already at
    # 4812 before this batch, Task 1 took it to 5006): 5006 -> 5185 (+179).
    # The two authoring forms' draft seams: the baseline reader, the one
    # dirty predicate every exit reads, the veto toast, the transition-only
    # resync helper and the five ``Changed`` recorders (+~150 with their
    # docstrings), the draft seeding/clearing beside every flag write, and
    # the per-item drop at the reader-row select seam.
    # 2026-10-21, TASK-34000.22 fix round 1 (same owner decision): 5185 ->
    # 5227 (+42): the central guard at the reader-row select seam, the
    # refusal's tab reveal, and the three review-set openers' refusals.
    # 2026-10-21, TASK-34000.22 fix round 2 (same owner decision): 5227 ->
    # 5254 (+27): the viewer opener's refusal, the central guard moved ahead
    # of the select preamble, and ``_reveal_library_media_dirty_form``.
    # 2026-10-21, TASK-34000.22 fix round 3 (same owner decision): 5254 ->
    # 5265 (+11): the reveal flips only from "list" (comment) and the
    # external-detail opener's one-line refusal.
    # 2026-10-22, P1-A final review C-1 (same owner decision): 5265 -> 5281
    # (+16): ``reader_action_target`` takes no view (the Reader's controls
    # act on the loaded item in the list view too), the highlight hold also
    # holds while a form is dirty in any view, and the trash confirm's title
    # derives from the same target id as the Delete press. Exact measurement.
    # 2026-10-27, TASK-34000.40 polish-1b Task 9 (N-33; owner decision --
    # expand limits, don't contort code): the analysis form's Save body moved
    # into ``_save_library_media_analysis_from_form`` so the press and Ctrl+S
    # share it: 5281 -> 5286 (measured with ``_measure``).
    # 2026-10-27, Task 9 fix round 1 (review I-3; same owner decision): the
    # analysis save's in-flight guard -- a second press says "Already
    # saving." and writes nothing; the worker clears the flag in a
    # ``finally`` (``_run_library_media_analysis_form_save``): 5286 -> 5313
    # (measured with ``_measure``).
    # 2026-10-27, Task 9 fix round 2 (review R1 N-6; same owner decision):
    # opening a row, an external detail and Discard each reset the analysis
    # save guard with the form: 5313 -> 5316 (measured with ``_measure``).
    # 2026-10-27, Task 9 fix round 3 (review R2 N2-2; same owner decision):
    # the analysis save guard is the token of the save that holds it, and a
    # worker clears it only while it still holds its own: 5316 -> 5321
    # (measured with ``_measure``).
    "tldw_chatbook/UI/Library_Modules/library_media_controller.py": 5321,
    # 2026-09-08, wave-8 task 2 (notes controller PR, notes series 2/N):
    # born governed. 185 moved methods carrying 3,934 source lines of body,
    # plus the module docstring, imports, the constructor's 93 keyword-only
    # parameters (the state accessor plus 92 named dependencies), the 102
    # binding properties and the generated state-shim loop.
    # (Erratum, review round: this comment shipped as "the 93-dependency
    # constructor, the 103 binding properties". 92 is the dependency count
    # and 102 the property count -- both measured off the class body; the
    # fix round that removed one dependency and one property corrected the
    # wiring-test pin and this row's number, but not this sentence.)
    #
    # 2026-09-08, wave-8 task 3 (notes cleanup, notes series 3/3): 5254 ->
    # 5276 (+22), and every one of those lines is ONE new binding property
    # plus its docstring -- `_notes_state`, this cluster's own state object
    # under the screen's own name. No moved body was touched (a cleanup PR
    # prunes the SCREEN's delegators, never the controller's methods), and no
    # moved body spells `self._notes_state` either: they all read the flat
    # `_library_notes_<field>` properties from the generated shim loop, which
    # is what keeps them byte-for-byte. The accessor exists for the SHARED
    # `canvas_sync.py` dispatchers, which 26 of this cluster's methods hand a
    # bare `self`, so their dotted spellings have to resolve on EITHER
    # receiver. Mutation-verified: removing it reds
    # `test_notes_row_toggle_resolves_the_dotted_state_path[controller]` and
    # leaves the `[screen]` leg green.
    #
    # 2026-09-08, task-32052 AC#1 (critique-8 keyboard branch, PR #2533):
    # 5276 -> 5283 (+7), a BEHAVIOUR FIX, not a move. The surface-mismatch
    # guard at the top of `_restore_library_notes_after_targeted_sync` (5
    # lines of comment + 2 of code) landed on the SCREEN's copy of that
    # method before wave-8 moved the body here; the merge re-applied it at
    # the body's new home, which is the only place it can run. Pinned by
    # `Tests/UI/test_library_crit8_keyboard.py::
    # test_ctrl_n_into_new_note_also_focuses_blank_note` -- reverting the
    # guard reds it (focus lands on a notes-tree row, not Blank note).
    #
    # 2026-09-09, task-32129 (Obsidian mode for Import once): 5284 -> 5300
    # (+16), a BEHAVIOUR ADD, not a move -- one new `@on` handler
    # (`handle_library_note_import_obsidian_mode`) that re-runs the read-only
    # check when the review's Obsidian-vault toggle changes. The row was
    # already 1 line under the file when this branch started (5284 measured
    # against a 5283 pin), so this re-pin also corrects that drift. Pinned by
    # `Tests/UI/test_library_note_import_flow.py::
    # test_obsidian_review_defaults_on_shows_skips_and_never_touches_the_vault`
    # -- without the handler the toggle press never re-checks and the test reds
    # on "Turning Obsidian vault off never re-ran the check."
    #
    # 2026-10-03, TASK-34000.2: 6366 -> 6311 (-55), a MOVE. The note
    # location loader's body and its file-written label moved into
    # `library_notes_sync_attention.py` (which now also answers whether the
    # note's sync folder is held for attention); the controller keeps a
    # three-line delegator and one scheduling call.
    #
    # 2026-10-08, TASK-34000.13: 6311 -> 6337 (+26), owner decision
    # (2026-10-04 ruling: a fix over a size ratchet raises the row to the
    # measured value; never move or reflow code to make it fit). The
    # delete prompt is now revealed after the refresh:
    # `handle_library_note_delete` schedules
    # `_reveal_library_note_delete_prompt` (scroll the whole prompt into
    # Info, focus Cancel in place) via `call_after_refresh`.
    #
    # 2026-10-09, TASK-34000.8: 6337 -> 6338 (+1), owner decision (same
    # ruling). Not this task's code: the TASK-34000.13 review commit
    # (8a8970dd2e, "blank line before _reveal_library_note_delete_prompt")
    # added one line after the 6337 pin was measured, so the row was red on
    # the base. Re-set to the exact measurement.
    #
    # 2026-10-09, TASK-34000.25: 6338 -> 6383 (+45), owner decision (same
    # 2026-10-04 ruling: a fix over a size ratchet raises the row to the
    # measured value; never move or reflow code to make it fit). This task's
    # code: the Media Reader's Note action creates its note through the one
    # create seam (`_start_library_note_from_media_source`, beside
    # `_start_library_blank_note`) and the caret lands where the latest
    # create asked (`_focus_library_note_caret_for_create`, read by both the
    # create projection and the editor-ready hook). Re-set to the exact
    # measurement.
    #
    # 2026-10-09, PR #3055 review (Minor 4): 6383 -> 6386 (+3), owner
    # decision (same 2026-10-04 ruling). Three blank lines between
    # consecutive method definitions (`_focus_library_note_caret_for_create`,
    # `_start_library_note_from_media_source`, `_focus_library_note_control`)
    # that the adapt batch left out. Re-set to the exact measurement.
    #
    # 2026-10-21, TASK-34000.9: 6386 -> 6389 (+3), owner decision (same
    # 2026-10-04 ruling). ``_library_note_presentation_state`` carries the
    # new ``autosave_scheduled`` flag (one kwarg line) read through a lazy
    # import of the autosave policy module (one line plus its blank); the
    # predicate itself is in ``library_note_autosave.py``. Re-set to the
    # exact measurement.
    #
    # 2026-10-21, TASK-34000.10: 6389 -> 6397 (+8), owner decision (same
    # 2026-10-04 ruling). ``_library_note_presentation_state`` carries the
    # vetoed field and its reason (``validation_field`` /
    # ``validation_message``: one ``vetoed`` line, two kwargs, a three-line
    # comment), and the two explicit veto presenters (Save, conflict
    # Overwrite) each record the field in one line; the focus-independent
    # guard itself is in ``library_note_autosave.py``, the tidy in
    # ``Library/library_note_field_tidy.py``. Re-set to the exact measurement.
    # Fix round 1 (2026-10-22): 6397 -> 6425 (+28), same ruling; fix round 2:
    # 6425 -> 6428 (+3), the mark checked before "Saving…" (three comment
    # lines; the check itself moved). The mark
    # clears only when the vetoed field itself changes (M-5): one
    # ``clear_vetoed_field`` line in each of the four field handlers, one in
    # ``_begin_library_note_load`` and one in ``_reset_library_note_editor_
    # state`` (M-6); the two explicit veto presenters pass the veto's message
    # (two lines each); ``safe_next_action`` is suppressed while a veto
    # stands (M-4: one kwarg, three comment lines); the presentation comment
    # on the ungated mark grew by two lines; the status line keeps the veto
    # reason while a mark stands and a conflict or a landed save clears it
    # (nine lines). Exact measurement.
    # 2026-10-22, P1-A final review I-1 (same owner decision): 6428 -> 6473
    # (+45): ``_library_notes_defer_focus_to_app_focus`` -- while the
    # terminal is blurred a focus-identity restore focuses nothing and
    # re-points Textual's ``_last_focused_on_app_blur`` at the live instance
    # of the user's field, so AppFocus lands there (docstring included), plus
    # the two-line call at the restore's ``set_focus``. Exact measurement.
    # 2026-10-22, TASK-34000.10 final-review re-review (owner decision, same
    # rule): 6473 -> 6484 -- the blurred-terminal restore leaves another
    # screen's or a modal's own focus slot alone.
    # 2026-10-24, TASK-34000 P1-B rebased onto dev with P1-C (owner decision,
    # "expand the limits, don't modify other code to make it fit"): the batch's
    # .17/.14/.12/.11/.71 controller work (Add-from-files reset, root-load
    # belt, link picker insert/follow, lazy-start settle) measured on the
    # rebased tree: 6484 -> 7200.
    # 2026-10-24, TASK-34000.28 (polish-1a task 2; same owner decision): the
    # folder-row Right/Left helper ``_library_notes_folder_row_arrow`` (one
    # method, docstring included), the Sort chooser's two focus hand-offs
    # (``then=`` pair, Skills shape), its strip-focus forwarding property and
    # the ``sort``/``sort-choice:`` role targets (a vanished choice resolves
    # to its opener): 7200 -> 7298 (measured).
    # 2026-10-25, TASK-34000.28 fix round 1 (same owner decision): the
    # expand/collapse step split out of the folder-row press handler
    # (``_toggle_library_notes_folder``) so Right/Left toggle without
    # selecting the folder: 7298 -> 7312 (measured).
    # 2026-10-25, TASK-34000.28 (polish-1a task 3; same owner decision): the
    # shared slice helper ``_ensure_library_notes_folder_slices`` (the toggle's
    # loop, now also the locator's) and the folder pickers' all-folders loader
    # ``_page_library_notes_folder_targets`` replacing the loaded-branches
    # ``_library_notes_folder_target_options``: 7312 -> 7339 (measured).
    # 2026-10-26, TASK-34000.28 (polish-1a task 4 and its fix round 1; same
    # owner decision): the folder-conflict notice's Refresh handler
    # (``handle_library_notes_notice_refresh``) and its focus role's
    # selector, and a successful delete Undo withdrawing that notice and its
    # Refresh offer, which used to outlive it until restart: 7339 -> 7362
    # (measured).
    # 2026-10-26, TASK-34000.28 (polish-1a task 5; same owner decision): the
    # one-hop way back after a followed link -- ``link_origin`` set by the
    # backlink and note-link opens and cleared by every other open in
    # ``_begin_library_note_load``, their shared flush-then-open tail
    # ``_open_library_note_from_link``, the ``‹ <title>`` handler
    # ``handle_library_note_link_back`` and its focus role's selector:
    # 7362 -> 7393 (measured). Fix round 1 (same decision): an editor exit
    # clears ``link_origin`` too (I-1): 7393 -> 7394 (measured).
    # 2026-10-26, TASK-34000.29 (polish-1b Task 6; owner decision "expand
    # limits, don't contort code"): ``_cancel_library_notes_operation``, the
    # operation's ``origin_control`` and the synchronous focus return after a
    # finish or cancel, the ``use-in-console`` focus role, and the one
    # per-note fact reset (``_reset_library_note_facts`` /
    # ``_start_library_note_fact_loads``) the open and create paths share:
    # 7394 -> 7470 (measured). Fix round 1 (same decision; review R0 M-3): the
    # deferred focus return is one programmatic, region-checked retry, and
    # the presentation state names the transfer's origin control:
    # 7470 -> 7478 (measured).
    # 2026-10-26, TASK-34000.29 (polish-1b Task 7, N-14; same owner
    # decision): select mode's Space action, the Actions… chooser and its
    # Escape, the bulk Move and bulk keyword seams with their receipts and
    # the one shared finish (``_finish_library_notes_bulk``), the open note's
    # reload after a keyword change, the chooser's focus roles, the
    # synchronous focus hand-offs and the in-place tree refresh
    # (``_refresh_library_notes_tree_in_place``): 7478 -> 8008 (measured).
    # 2026-10-27, Task 7 fix round 1 (review R0 M-1, M-8; same decision): the
    # owed visit reload (``_tree_reload_owed``, ``_release_library_notes_
    # mutation``, ``_run_owed_library_notes_tree_reload``) and the empty
    # keyword toast: 8008 -> 8044 (measured).
    # 2026-10-27, TASK-34000.29 (polish-1b Task 8, part 8a, N-29; same owner
    # decision): Recently deleted pages -- the page read with its read-side
    # clamp (``_load_library_notes_trash(page=)``, ``_read_library_notes_
    # trash_page``), the Previous/Next handlers queued behind a running
    # Restore (``_trash_reload_owed``), the Trash focus roles and their
    # resolution (``_library_notes_trash_focus_target``) and the owed landing
    # (``_trash_focus_role``): 8044 -> 8271 (measured).
    # 2026-10-27, Task 8 fix round 1 (review R0 m-1, m-3, m-6, m-7; same
    # decision): the Trash list as a scroll owner, a failed page read keeping
    # the painted page, page 1's loading state on a reopen, the owed landing's
    # lifetime comment: 8271 -> 8305 (measured).
    # 2026-10-27, Task 8 fix round 2 (review R1 N-1, N-2; same decision): a
    # page change replays no old offset (the Trash scroll owner matches the
    # page it shows), a failed read marks the snapshot and the Retry handler
    # re-runs it: 8305 -> 8358 (measured).
    # 2026-10-27, Task 8 fix round 3 (review R2-1..R2-3; same decision): a
    # failed read lands focus on Retry, a landed control is scrolled in after
    # the deferred offset re-applies (``_keep_landed_trash_control_in_view``),
    # a reopen clears the old failure: 8358 -> 8393 (measured).
    # 2026-10-27, Task 8 fix round 4 (review R3-1, ruling T8-j; same
    # decision): while a page change's landing settles, the Trash scroll
    # owner records no offset (``_trash_landing_settling``), so a sync on
    # either side of the keep-in-view cannot hide the landed control:
    # 8393 -> 8421 (measured).
    # 2026-10-27, TASK-34000.40 polish-1b Task 9 (N-33; same owner decision
    # -- expand limits, don't contort code): the Save press's body as
    # ``_begin_library_note_explicit_save``, which Ctrl+S runs too:
    # 8421 -> 8425 (measured).
    # 2026-10-27, Task 9 fix round 1 (ruling T9-g; same owner decision): a
    # clean note's commit answers "Saved — no changes." on the status line
    # for the key and the button alike: 8425 -> 8432 (measured).
    # 2026-10-27, Task 9 fix round 2 (review R1 N-3; same decision): only an
    # explicit Save is answered "Saved — no changes." -- the outcome learns
    # who asked: 8432 -> 8441 (measured).
    # 2026-10-29, TASK-34000.40 Task 10 (N-34; owner decision "expand limits,
    # don't contort code"): Escape leaves a running import instead of
    # cancelling it (a running check still cancels, ruling T10-c), and the
    # lasting-sync snapshot is handed to the canvas with its back control
    # named for the route in ("‹ Sync folders"): 8441 -> 8462 (measured).
    # Same day, same decision: that hand-off is class-forwarded, because the
    # canvas-kwargs tests run the method on a fake: 8462 -> 8467 (measured).
    # 2026-10-29, Task 10 fix round 1 (review R0 I-1/M-7; same decision):
    # one "back to Sync folders" decision read by Escape, the back control,
    # the label and the footer, a shared roots return that lands focus, and a
    # running-import re-entry focused on its progress: 8467 -> 8505 (measured).
    # 2026-10-29, TASK-34000.29 Task 8b (N-29, Delete forever; owner ruling
    # "yes with confimration", owner decision "expand limits, don't contort
    # code"): the Delete forever handler and its ConfirmationDialog, the
    # confirm and cancel answers (focus owed back to the row), the purge
    # worker through ``NotesScopeService.purge_note`` with its refusals, its
    # notice and its page re-read after the interlock's release, and the
    # parked R4-1/R4-2 settling rules (every list landing raises the marker,
    # a landing for another role drops it, keep-in-view waits out a pruning
    # sync's framework focus): 8505 -> 8739 (measured).
    # 2026-10-29, TASK-34000.29 Task 8b fix round 1 (review R0; same owner
    # decision): Delete forever's busy reason painted on the Trash and
    # cleared at the interlock's release (m-5), one confirm at a time with a
    # stale-confirm recovery (m-8), the destructive confirm's class (m-4),
    # the page read's "what stays elsewhere" facts for the confirm (m-9), and
    # a roleless read dropping a replaced landing's marker (R4 residual):
    # 8739 -> 8845 (measured).
    # 2026-10-29, TASK-34000.29 polish-1b final review I-2/I-3/m-3 (same owner
    # decision): the Trash's reads (opener, Previous/Next, Retry) wait behind
    # the writer interlock alone (``_library_notes_writer_held``), never an
    # import's fence; the ONE release body (every site, the bulk workers'
    # ``finally`` and the import's settle path through it) runs the owed
    # Trash read and clears the busy reason; leaving the Trash drops a queued
    # read: 8845 -> 8893 (measured).
    # 2026-10-29, final review T6 m2 (same owner decision): the origin focus
    # return's docstring names it a backstop while the origin stays enabled:
    # 8893 -> 8896 (measured).
    "tldw_chatbook/UI/Library_Modules/library_notes_controller.py": 8896,
    # See the dev-side-controller note above the character-repair row. Dev
    # landed this file at 195 lines; the +3 is this merge's own port -- the
    # `apply_navigation_context` gate read the flat `_library_prompts_
    # mutation_in_flight` attribute the wave-6 prompts cleanup deleted, and
    # was retargeted to `_prompts_state.mutation_in_flight` with a one-line
    # comment naming the retarget (3 comment lines, the gate line itself
    # replaced in place).
    "tldw_chatbook/UI/Library_Modules/library_navigation_controller.py": 202,
    "tldw_chatbook/UI/Library_Modules/library_media_trash_browse_controller.py": 319,
    # 2026-09-09, task-32129 (Obsidian mode for Import once): 587 -> 602
    # (+15), a BEHAVIOUR ADD -- `set_obsidian_mode` (6 lines) plus the
    # vault-detection flag and the two `obsidian_mode` arguments threaded
    # through `_plan_selection`/`check`.
    "tldw_chatbook/UI/Library_Modules/library_note_import_controller.py": 793,
    # 2026-09-11, task-32243 (refused Check names its reason): 2023 -> 2024
    # (+1), and that one line is the `check_failure_line` import. The refusal
    # copy table and its two helpers live in
    # `Library/library_notes_lasting_sync_state.py` with the rest of this
    # screen's presentation state; both fallback strings moved into the helper
    # (keyed on `root_id == ""` => setup), so each of the two Check paths
    # spends exactly one line on it. This row was briefly pinned at 2128 with
    # the table in the controller; that was rejected in review and reverted.
    #
    # 2026-10-03, TASK-32633 slice (N-03): 2380 -> 2366 (-14), a MOVE. The
    # root-status copy table moved to `Library/library_notes_lasting_sync_
    # state.py` as `ROOT_STATUS_LABELS`, next to the refusal table above, and
    # the healthy label is now dated there (`root_status_label`); the
    # controller spends one line on the call it used to spend three on.
    #
    # 2026-10-08, TASK-34000.50, owner decision ("expand the limits, don't
    # modify other code to make it fit", 2026-10-04): 2366 -> 2375 (+9). The
    # third per-root label overlay (a healthy root nothing watches reads
    # "Sync stopped") sits inline beside the other two in `_project_root`.
    #
    # 2026-10-09, TASK-32451, owner decision ("expand the limits, don't
    # modify other code to make it fit", 2026-10-04): 2375 -> 2378 (+3). The
    # row title is the runtime's `display_name` (the folder's name) instead
    # of a placeholder literal; three comment lines name the source.
    # 2026-10-20, TASK-32451 rebase onto dev 250159834c, same owner decision:
    # 2378 -> 2397. Dev itself already measures 2394 against its own 2375
    # row (a pre-existing dev red); this branch adds the same +3 above.
    #
    # 2026-10-24, TASK-34000.16 fix round 1 (I-2), owner decision ("expand
    # the limits, don't modify other code to make it fit", 2026-10-04):
    # 2397 -> 2415 (+18). A migrated candidate whose file the walk could not
    # read holds Activate: `_review_projection` hands the review's `source`
    # to the builder (+1), the migration check's status line names the
    # skipped files (+4), and `activate_root` refuses before asking the
    # runtime (+10 with its comment); the import is one line.
    # 2026-10-24, TASK-34000 P1-B rebased onto dev with P1-C (same owner
    # decision): P1-B's .17 begin_add_from_files / release-on-exit and .71
    # availability re-derive on top of P1-C's .16 activation hold, measured
    # on the rebased tree: 2415 -> 2461.
    # 2026-10-29, TASK-34000.40 Task 10 (N-34, AC#6; owner decision "expand
    # limits, don't contort code"): staged review choices survive leaving the
    # review -- forgotten per root by an apply, a stale review or a root
    # control, bounded to three observations -- and the status line is
    # derived from the staged count: 2461 -> 2524 (measured).
    # 2026-10-29, Task 10 fix round 1 (review R0 M-2/M-3; same decision): the
    # inert root action keeps staged choices, and one status-line derivation
    # puts a held Activate's reason ahead of them: 2524 -> 2548 (measured).
    "tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py": 2548,
    "tldw_chatbook/UI/Library_Modules/library_prompt_browse_controller.py": 281,
    # 2026-09-05, wave-6 task 2 (prompts controller PR, series 2/3): born
    # governed the moment this file existed (task-31203 AC#4's glob-based
    # discovery, recipe §17) -- 139 moved methods (byte-for-byte; the
    # largest single move of this program, past skills' own 86) plus the
    # constructor/property scaffolding the canon requires, pinned at its
    # exact measured line count. Constructor arity MEASURED with
    # `inspect.signature(LibraryPromptsController.__init__)`, never
    # hand-counted: 33 parameters including `self` -- 1 positional
    # (`screen`) + 31 keyword-only named dependencies (1 state accessor +
    # 12 shell helpers + 4 shared-shell-state accessors + 3 prompt-wiring-
    # controller accessors + 1 merely-delegate-property accessor + 10
    # late-binding callables for the exclusions). 85 class-level
    # `property` objects: 42 hand-written bindings + the 43 generated
    # flat-name state shims. See the module's own docstring for the full
    # 161-candidate / 22-exclusion derivation and the single-controller
    # decision (one connected component of 145 names, no seam to split on).
    #
    # 2026-09-05, wave-6 task 2 fix round 2 (post-review, counts only):
    # comment-only growth, no method body touched (139 movers unchanged, all
    # still byte-for-byte). Three census figures in this module's own
    # docstring were wrong and were corrected in place: "Three MORE names"
    # reach the unbound-fake-self shape by indirection -> FIVE (the sentence
    # was counting the three SHAPES as if they were names; the bullets always
    # listed 1 + 2 + 2, and 10 direct + 5 indirect = the 15 rows the same
    # paragraph enumerates); "11 movers" forward bare `self` into `_sync_
    # library_canvas` -> 7 movers + 4 exclusions (AST re-scan: 11 methods
    # total, `set & movers` = 7); and the `_sync_library_canvas` LATENT
    # verdict's supporting evidence, which claimed only one test function
    # mentions a mover -- false, `test_library_entry_compose_once.py` INVOKES
    # `_sync_library_prompts_browse_result` at :1014/:1044. The VERDICT is
    # unchanged and now rests on the correct argument (monkeypatch is
    # function-scoped; that file's four patch pairs live in four OTHER test
    # functions, zero overlap, verified by mapping every census line and both
    # invocation lines to their enclosing FunctionDef). The "site" definition
    # and the alternative un-deduplicated count (42) are now stated inline so
    # a reviewer can re-derive 33 without re-running the census.
    # 4956 -> 4991.
    # 2026-09-05, wave-6 task 3 (prompts cleanup PR, prompts series 3/3):
    # comment-only growth, 4991 -> 4998. Two now-false present-tense claims
    # about the screen's delegators ("keeps one-line delegators under every
    # one of these 139 original names", module docstring, and the same claim
    # again in `LibraryPromptsController`'s own class docstring) were
    # corrected to the post-prune 100-of-139 count and pointed at
    # `_PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED` -- the identical shape the
    # skills and ingest cleanups each had to fix in their own controllers.
    # No moved body was touched (byte-for-byte canon intact); this is the
    # §17 re-pin-at-move flow applied to a docstring-only delta. 4991 ->
    # 4998.
    # TASK-36454..36458: narrow save/install completion helpers reduce this owner.
    # 2026-10-27, TASK-34000.40 polish-1b Task 9 (N-33; owner decision --
    # expand limits, don't contort code): the Save handler's body as
    # ``_save_library_prompt_from_editor`` (shared with Ctrl+S, which also
    # reveals the receipt below the fold), Ctrl+S's own seam
    # (``_save_library_prompt_from_key``: a held Save says why, a conflict
    # shows its banner, a clean editor says nothing was pending) and the
    # canvas-backed refusal reading: 5100 -> 5161 (measured with ``_measure``).
    # 2026-10-27, Task 9 fix round 1 (ruling T9-j; same owner decision): the
    # footer re-registers when a write starts and ends, so the Ctrl+S chip
    # never shows while Save is held: 5161 -> 5164 (measured with
    # ``_measure``).
    "tldw_chatbook/UI/Library_Modules/library_prompts_controller.py": 5164,
    # 2026-09-03, wave-3 task 3 (combined search+RAG controller PR, series
    # 2/3): born-governed by the glob above -- new file, pinned at its
    # exact measured line count on landing (42 moved methods + the
    # constructor/property scaffolding the byte-for-byte canon requires;
    # a 43rd candidate, `_load_library_search_history`, was excluded
    # mid-task -- module-globals coupling on a bare `get_cli_setting`
    # reference broke a real test fixture -- and stays on the screen).
    # 2026-09-03, wave-3 task 3 fix round 1: two false-caller-count claims
    # in this module's own docstring were corrected in place (no methods
    # touched, no bodies re-shaped -- byte-for-byte canon on the 42 moved
    # bodies is unaffected). 1857 -> 1890.
    # 2026-09-03, wave-3 task 4 (search+RAG cleanup, series 3/3): the ruled
    # cleanup item from task 3's own report -- `_sync_library_rag_scope_
    # toggle_and_run_gate_widgets`'s moved-body docstring carried the
    # ORIGINAL false caller claim ("Called synchronously from
    # `_apply_local_source_snapshot`'s in-place branch"), byte-for-byte
    # original text the byte-for-byte canon correctly forbade fixing in
    # task 3 itself -- corrected in place here to name the actual caller
    # (`_reconcile_library_entry_state`, screen-resident), matching the
    # module docstring's own already-corrected paragraph. Comment-only
    # growth; no method body, mover count, or byte-for-byte canon content
    # changed. 1890 -> 1895.
    # 2026-09-03, wave-3 task 5 (wave close, stale-doc sweep): this
    # module's own docstring claimed `LibraryScreen` "carries (task 2)"
    # the two-prefix shim in present tense -- stale since task 4 deleted
    # that shim. Corrected to past tense ("task 2 installed ... deleted
    # at cleanup, task 4"), matching `library_export_controller.py`'s own
    # correct precedent. Comment-only growth (+2 lines); no method body
    # touched. 1895 -> 1897.
    # 2026-09-03, wave-3 final review fix wave: the generated shim block's
    # OWN footer comment (near the bottom of the file, not the module
    # docstring task 5 already fixed above) still said "the shim block
    # `LibraryScreen` carries (task 2)" in present tense -- the same task-4
    # deletion task 5 accounted for at the top of the file, missed at the
    # bottom. Reworded to match the module docstring's corrected past-tense
    # phrasing. Comment-only growth (+1 line); no method body touched.
    # 1897 -> 1898.
    # 2026-10-19, TASK-34000.24 (owner decision 2026-10-04, same rule): 1923 ->
    # 2028 (+105). Search ▸ evidence ▸ `u` now obeys the readers' link-on-use
    # rule: `_link_library_rag_result_on_use` (resolves the row-model id,
    # links through `Workspaces/link_on_use`, refuses a non-linkable block)
    # plus the two injected dependencies it needs (the workspace depth
    # state and its invalidation) with their properties and Args entries.
    # 2026-10-20, TASK-34000.24 fix rounds 1-2 (owner decision 2026-10-04,
    # same rule): 2028 -> 2051. Fix round 1 added +17 (the seam's one
    # decision call, the receipt said after the stage, the press's own
    # write taken back on a refused stage) -- the row raise that round's
    # commit missed; fix round 2 added +6 (one plain sentence whether the
    # opener refused or raised).
    # 2026-10-20, TASK-34000.24 final-review fix (F7; owner decision
    # 2026-10-04, same rule): 2051 -> 2056 (+5). The locked-Console "staged"
    # notice is read before the opener but sent only after it succeeds, so
    # a refused stage no longer shows "staged" beside "could not take it".
    "tldw_chatbook/UI/Library_Modules/library_rag_search_controller.py": 2056,
    "tldw_chatbook/UI/Library_Modules/library_skill_import_controller.py": 805,
    # 2026-09-25, TASK-32954 (built-in skills): +3 lines threading the
    # Library-only `include_disabled_builtins` flag through the browse read so
    # a disabled built-in stays listed and re-enableable. 413 -> 416.
    "tldw_chatbook/UI/Library_Modules/library_skills_browse_controller.py": 416,
    # 2026-09-04, wave-4 task 2 (skills controller PR, series 2/3): born
    # governed the moment this file existed (task-31203 AC#4's glob-based
    # discovery, recipe §17) -- 86 moved methods (byte-for-byte) + a
    # 40-parameter constructor (self + screen + 38 named dependencies; the
    # largest single move of the effort; see the module's own docstring
    # for the full derivation) pinned at its exact measured line count.
    # 3181 -> 3113 -> 3099 (two fix rounds: 5 methods total reverted to
    # screen-resident after the battery caught bare-self identity-
    # comparison regressions -- `_library_screen_is_current(self)` in
    # four Import-row handlers, and `self.app.screen is self` inline in
    # `_present_library_skills_import_snapshot`; see the module's own
    # docstring, exclusion 5) -> 3131 (post-landing-review fix: a SIXTH
    # bare-self hazard, an unbound-attribute escape via `getattr(self,
    # "focused", None)` with no corresponding property, silently degraded
    # the committed-mutation-refresh focus-restore path; fixed by adding
    # the `focused` framework-service property every sibling controller
    # already carries -- see the module's own docstring, exclusion 5).
    #
    # 2026-09-04, wave-4 task 3 (skills cleanup, series 3/3): comment-only
    # growth, no method body touched (86 movers unchanged -- cleanup prunes
    # the SCREEN's delegators, not the controller's own methods, per the
    # collections/search+RAG precedent). Two moved-docstring-adjacent module
    # docstring corrections: the "6-match gap is three @property/@x.setter
    # pairs" arithmetic error (should be SIX -- 2 raw defs - 1 unique name =
    # 1 gap per name, 6 names = 6 gap; the same error task 2's own report
    # caught and fixed in its own text (§12c) but missed here) and the
    # "LibraryScreen keeps one-line delegators under every one of these
    # original names" claim, now false for 16 of the 86 (this task's own
    # screen-side prune) -- both fixed in this task, +9 lines. 3131 -> 3140.
    #
    # 2026-09-04, wave-4 final review (fix wave): two comment/import-only
    # changes, no method body touched. (1) Removed the dead
    # `LIBRARY_SKILLS_IMPORT_WORKER_GROUP` import from `.screen_constants`
    # (zero in-file uses -- the screen, not this controller, is the one
    # consumer of that name), -1 line. (2) Reworded the `focused`-property
    # fix-round docstring paragraph to align its "SIXTH ... distinct in
    # shape" framing with recipe §3's landed "sixth bypass shape, close
    # cousin" framing (the getattr/focused escape is that shape's own
    # sub-case -- the seventh instance counted under it -- not an eighth/
    # new shape), +3 lines. Net 3140 -> 3142.
    "tldw_chatbook/UI/Library_Modules/library_skills_controller.py": 3142,
}

#: Loose on purpose (see `test_screen_size_ratchet.py`'s own 200-line
#: tolerance for the same reasoning at a larger scale): ordinary in-file
#: edits — a strengthened comment, a docstring fix — must not fail this
#: check. It exists only to catch a budget left stale-high after a
#: controller shrinks (a cleanup PR that forgot to lower its row), not to
#: pin every file to its exact byte count at all times.
_SLACK_TOLERANCE_LINES = 50

# TASK-31244: initial exact pin for the focused navigation helper. Existing
# controller ceilings remain unchanged; route hooks stay outside LibraryScreen.
#
# 2026-09-07, wave-7 final review: `origin/dev` reconciliation merge. Dev
# landed this file at 811; the +6 is this merge's own port and nothing else.
# Dev moved `_apply_navigation_context_state`'s whole body out of
# `LibraryScreen` and into this module, carrying two writes to the flat
# `_selected_media_id` / `_library_media_view` attributes the media cleanup
# PR deleted. Left alone those were SILENT stray attributes -- writes, not
# reads -- so every "open in Library" media deep link would have painted the
# wrong surface with no exception anywhere. Retargeted to
# `_media_state.selected_media_id` / `.view` with a six-line comment naming
# the retarget (the two lines themselves are replaced in place). 811 -> 817.
#
# 2026-10-26, TASK-34000.28 (polish-1a task 5 fix round 1; owner decision
# 2026-10-04, expand-the-limits rule): the unmounted note deep link is a plain
# open, so it clears ``link_origin`` like every other (I-1): 840 -> 841
# (measured).
_BUDGETS["tldw_chatbook/UI/Library_Modules/library_unavailable_navigation.py"] = 841


# ADR-172 / TASK-32870–32872: initial exact pins for the new artifact owners.
# PR #2754 reviews: initial pin includes finite workers, reader reveal, typed builders and normalized exports.
# 2026-10-09, TASK-34000.23: 806 -> 876 (+70), owner decision ("expand the
# limits, don't contort code", 2026-10-04). The Reports demo CTA now asks
# before seeding: `DemoConsent` (the block's state), `open_demo_consent`
# (resolves the billed provider · model through
# `resolve_persisted_briefing_defaults`, reads `list_briefing_schedules`
# for the reuse wording, no-provider -> disabled run button),
# `close_demo_consent`, and `start_demo` (the old no-confirm press, now the
# only way in). Nothing was moved out to make it fit.
# 2026-10-09, TASK-34000.23 review 1: 876 -> 909 (+33), same owner decision.
# The 876 pin was measured before the round's last two lines landed (878 at
# the head commit, review Important #1); this round adds
# `_reused_schedule_pair` (a reused schedule's default preset bills its own
# provider/model, so the consent names that pair) and the schedule read it
# hangs off. Measured with `len(read_text().splitlines())` after the edits.
# 2026-10-20, TASK-34000.23 rebase onto dev 250159834c, same owner decision:
# 909 -> 1020. Dev itself already measures 904 against its own 806 row (a
# pre-existing dev red); this branch adds the same +116 the rows above
# record (the demo consent block). Measured with `len(read_text().splitlines())`.
# 2026-10-20, TASK-34000.23 final-review fix (F4; owner decision 2026-10-04,
# same rule): 1020 -> 1057 (+37). `empty_actions_visible`, the ONE predicate
# for the Reports empty-state actions (empty page, or a failed newest first
# run), moved out of the items pane's `sync` so the work pane can hide its
# standing "Watchlists" button while the items pane offers the empty-state
# one; plus the reused schedule's name handed to the consent copy (F3).
# Measured with `len(read_text().splitlines())`.
_BUDGETS["tldw_chatbook/UI/Library_Modules/library_artifacts_controller.py"] = 1057
_BUDGETS["tldw_chatbook/UI/Library_Modules/library_artifacts_share_controller.py"] = 339

# 2026-09-25, TASK-32954 (ruling R14): initial exact pin for the built-in
# skills cluster (read-only preview, Customize, Enabled switch, review-link
# route), split out so `library_skills_controller.py` stays within its budget.
_BUDGETS["tldw_chatbook/UI/Library_Modules/library_skills_builtin_controller.py"] = 253
# 2026-09-26, TASK-32954 Qodo review #2 (PR #2842, pre-merge): +3 so Customize
# says a tampered built-in failed its integrity check instead of blaming an
# existing folder.
_BUDGETS["tldw_chatbook/UI/Library_Modules/library_skills_builtin_controller.py"] = 256


@lru_cache(maxsize=None)
def _measure(rel_path: str) -> int:
    """Line count of a `Library_Modules` controller module.

    Cached: the ceiling and anti-slack tests both run over the same
    parametrization, so an uncached version reads every budgeted file
    twice per session for no extra coverage. The cache is per-process, so
    mutation-testing this ratchet (edit the file, re-run pytest as a fresh
    process) still works — just not by editing a file from inside a
    running test, the same caveat `test_screen_size_ratchet.py._measure`
    documents.

    Args:
        rel_path: Repo-relative path to the module.

    Returns:
        int: line count via `str.splitlines()`.

    Raises:
        AssertionError: If the module is missing — the budget entry is
            stale and must be updated deliberately, not silently skipped.
    """
    path = _REPO_ROOT / rel_path
    assert path.exists(), f"{rel_path} not found; the budget entry is stale."
    return len(path.read_text(encoding="utf-8").splitlines())


def _discovered_controller_paths() -> list[str]:
    """Every `*_controller.py` under `Library_Modules/`, repo-relative posix paths."""
    return sorted(
        p.relative_to(_REPO_ROOT).as_posix()
        for p in _LIBRARY_MODULES_DIR.glob(_CONTROLLER_GLOB)
    )


@pytest.mark.unit
def test_every_controller_file_has_a_budget_row() -> None:
    """The self-defending property: an unlisted controller fails loudly.

    Unlike `test_screen_size_ratchet.py`'s hand-maintained `_BUDGETS` (no
    check that every screen file has a row), this test globs the
    directory at collection time, so a NEW `*_controller.py` — wave 3's
    search/RAG controller(s), or any of the six subsystems after it —
    fails here by name instead of landing ungoverned.
    """
    discovered = _discovered_controller_paths()
    missing = [rel_path for rel_path in discovered if rel_path not in _BUDGETS]
    assert not missing, (
        "New Library_Modules controller file(s) with no _BUDGETS row: "
        f"{missing}.\n\n"
        "Add a row to _BUDGETS in "
        "Tests/Architecture/test_library_modules_size_ratchet.py, pinned "
        "at the file's current exact line count "
        "(len(path.read_text(encoding='utf-8').splitlines())) -- do not "
        "pick a round number 'for headroom.' See this module's docstring "
        "for the full governance rationale."
    )


@pytest.mark.unit
@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))
def test_controller_does_not_grow_past_its_budget(rel_path: str) -> None:
    """The ceiling itself: a budgeted controller may not exceed its pin.

    Args:
        rel_path: Repo-relative path of the budgeted module, supplied by
            the parametrization over `_BUDGETS`.

    Raises:
        AssertionError: If the module grew past its line budget without
            the budget being re-pinned in the same diff.
    """
    max_lines = _BUDGETS[rel_path]
    lines = _measure(rel_path)

    guidance = (
        f"\n\n{rel_path} is under a size ratchet "
        f"(Tests/Architecture/test_library_modules_size_ratchet.py).\n"
        f"If this growth is a sanctioned move landing (recipe "
        f"backlog/docs/library-decomposition-recipe.md §1), re-measure "
        f"and raise this row's pin in the SAME commit with a dated "
        f"comment. Otherwise this is the creep this test exists to "
        f"catch -- do not raise the budget just to silence it."
    )

    assert lines <= max_lines, (
        f"{rel_path} grew to {lines} lines (budget {max_lines}, "
        f"+{lines - max_lines}).{guidance}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))
def test_budget_is_not_left_slack_after_a_move(rel_path: str) -> None:
    """The recorded budget should track reality, not drift above it.

    A ratchet with slack silently permits regrowth up to the stale number,
    so a move that forgets to lower/tighten its budget after landing (or a
    ceiling raised more generously than the move that justified it) quietly
    buys the next feature headroom it was never meant to have.
    """
    max_lines = _BUDGETS[rel_path]
    lines = _measure(rel_path)

    assert max_lines - lines <= _SLACK_TOLERANCE_LINES, (
        f"{rel_path} is {max_lines - lines} lines under its budget "
        f"({lines} vs {max_lines}). Set it to {lines} so the real "
        f"measurement is what's pinned."
    )
