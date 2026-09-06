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
    "tldw_chatbook/UI/Library_Modules/library_character_repair_controller.py": 502,
    "tldw_chatbook/UI/Library_Modules/library_collections_capture_controller.py": 699,
    "tldw_chatbook/UI/Library_Modules/library_collections_controller.py": 1689,
    "tldw_chatbook/UI/Library_Modules/library_conversation_reader_controller.py": 943,
    "tldw_chatbook/UI/Library_Modules/library_conversations_controller.py": 1738,
    "tldw_chatbook/UI/Library_Modules/library_export_controller.py": 1307,
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
    "tldw_chatbook/UI/Library_Modules/library_ingest_controller.py": 2721,
    "tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py": 371,
    # See the dev-side-controller note above the character-repair row. Dev
    # landed this file at 195 lines; the +3 is this merge's own port -- the
    # `apply_navigation_context` gate read the flat `_library_prompts_
    # mutation_in_flight` attribute the wave-6 prompts cleanup deleted, and
    # was retargeted to `_prompts_state.mutation_in_flight` with a one-line
    # comment naming the retarget (3 comment lines, the gate line itself
    # replaced in place).
    "tldw_chatbook/UI/Library_Modules/library_navigation_controller.py": 198,
    "tldw_chatbook/UI/Library_Modules/library_media_trash_browse_controller.py": 319,
    "tldw_chatbook/UI/Library_Modules/library_note_import_controller.py": 587,
    "tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py": 2023,
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
    "tldw_chatbook/UI/Library_Modules/library_prompts_controller.py": 4998,
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
    "tldw_chatbook/UI/Library_Modules/library_rag_search_controller.py": 1898,
    "tldw_chatbook/UI/Library_Modules/library_skill_import_controller.py": 760,
    "tldw_chatbook/UI/Library_Modules/library_skills_browse_controller.py": 413,
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
_BUDGETS["tldw_chatbook/UI/Library_Modules/library_unavailable_navigation.py"] = 713


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
