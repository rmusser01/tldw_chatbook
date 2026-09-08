# Wave-7 Task 4 — Wave close (media series)

**Status:** COMPLETE.

**Branch:** `refactor/library-decomp-wave7-media`
**Wave start (parent):** `83e17323e`
**Wave tip at task open:** `014f695fc`

**Commits (rev-parse only, never transcribed):** see §11.

Deliverables: recipe trajectory + lessons (A), two follow-up filings with a
full id sweep (B), a stale-doc sweep (C), the full close battery incl. paired
baselines and an order-swapped probe (D), durable SDD evidence (E).

---

## 1. Pin trajectory — re-derived per COMMIT, and it corrected two published numbers

Re-derived by `git show <commit>:<path>` at **every one of the 21 commits in
`83e17323e..HEAD` plus the base itself (22 refs)**, reading the `_BUDGETS`
row in `Tests/Architecture/test_screen_size_ratchet.py` and the
`library_media_controller.py` row in
`Tests/Architecture/test_library_modules_size_ratchet.py`. Nothing was
carried over from a report. The full table is now in the recipe (§22's
"Wave-7 close"); the two findings that matter:

**Finding 1 — the controller was born at 4474, not 4461.** Task 2's report
§8 says "born **4461**", and the recipe's §22 table said the same until this
close. 4461 is the value AFTER fix round 1; the commit that first created
the row (`9187bd281`) reads **4474**. Neither commit is amendable —
`9187bd281` and `40fe1f5bc` are both recorded in `.git-blame-ignore-revs` by
hash — so the correction of record is in the recipe (§22 body and table) and
here.

**Finding 2 — the screen pin went DOWN and then back UP inside one task,
the only time in this program.** Task 2's GREEN pinned **34717/1282**; its
fix round 1 excluded `_exit_library_media_viewer` (recipe §3's new Form E,
callback identity) and put 37 lines BACK on the screen, re-pinning to
**34754/1282**. A task-level summary that reports only the endpoints
(`37333 → 34754`) is not wrong, but it hides the shape — and the shape is
exactly what a battery-found exclusion looks like in the ratchet. **This is
the argument for a wave close re-deriving per COMMIT rather than per task**,
and it is the only reason either of these two findings surfaced.

Full chains:

* screen `37537/1282 → 37333/1282 → 34717/1282 → 34754/1282 → 34669/1260`
* controller born `4474 → 4461 → 4496`

**Net wave-7 shrink: 2,868 screen lines and 22 methods** — second-largest
single-wave reduction in the program, behind prompts (3,819) and ahead of
skills (2,070), wave-2 (1,554, two subsystems), ingest (1,480) and
search+RAG (1,037). The 22 methods are exactly the 22 pruned delegators.

**Fresh `_measure()` at this close: 34669 / 1260 (screen), 4496
(controller)** — exact match to both pins, zero drift, nothing to lower.
This close's own tracked edits are comment/docstring-only; the two inside
the governed controller file were written LINE-NEUTRAL for that reason, and
`_measure()` re-reads 4496 afterwards, so no same-commit re-pin was needed.

---

## 2. Recipe updates (deliverable A)

All in `backlog/docs/library-decomposition-recipe.md`.

### 2.1 §3 — the screen-identity catalogue grows from three forms to five

The media series found two shapes the skills series' Forms A/B/C could not
see, and both are now written up as first-class forms with their incidents:

- **Form D — screen identity in a MEMBERSHIP test** (`self in
  <widget>.ancestors`). The existing census looks for `ast.Compare` with
  `Is`/`IsNot` plus bare `self` as a call ARGUMENT; this is `In`/`NotIn`, so
  a first pass over all 251 candidates returned a confident **zero**. The
  move shipped into the battery and produced 19 failed / 144 passed in the
  return-settlement cluster against a 2 failed / 161 passed isolated
  baseline.
- **The generalized census, recorded as the standing replacement for
  enumerating operators**: every bare `self` `ast.Name` in a moved body that
  is not the receiver of an attribute access. Exhaustive by construction.
  Both figures are recorded — **10** occurrences when run at the moment of
  the finding, **7** over the final mover set — because quoting only the
  survivor count makes an instrument that fired look like one that found
  nothing.
- **The 3-vs-4 corollary**: media carries FOUR membership sites, not the
  three it had to newly exclude (the fourth was already excluded on other
  grounds). Counting only newly-forced reclassifications understates the
  shape's frequency by 25%; count over ALL candidates.
- **Form E — CALLBACK identity**, where the method to exclude is the CALLER,
  not the callee (`self.call_next(self._apply_library_media_list_return,
  None)` vs an assertion comparing against `screen._apply_library_media_
  list_return`). Verified live at `Tests/UI/test_screen_navigation.py:3109`.

The list header now reads "FIVE exclusion-worthy forms" and points at Form
D's paragraph for the census that subsumes all five.

### 2.2 §3 — the fifth census spelling, verified as a RUNNABLE census

Read rather than rewritten, per the brief. The entry (written by task 3)
already states the shape (`ast.JoinedStr` composing a flat name), the
mechanism of the silent production defect, the runnable census ("walk
`ast.JoinedStr` nodes … over `tldw_chatbook/` and all of `Tests/`, keep
those whose literal fragments could compose one of the subsystem's flat
names"), both hits with their verdicts, the guard that was added, and the
mutation evidence both ways. **Re-verified against the live tree at this
close**: `canvas_sync.py:219` still builds
`f"_library_{kind}_row_selection"`, the dotted media branch is at `:217`,
`test_library_media_return_settlement.py:1745` still builds the method-name
f-string, and `test_library_selection_updates.py:343` is the added guard.
No polish needed.

### 2.3 §3 — exclusion-justification honesty (new standing rule)

Placed at the head of §3, beside the governing monkeypatch rule it is about.
Incident: the media controller's docstring shipped a draft justifying all 16
instance-attribute-monkeypatch exclusions with the *narrower* condition ("a
mover calls it, so the patch is bypassable"). A caller map re-derived at the
parent showed that true for **9** and FALSE for **7**. The exclusion SET was
correct; only the stated reason was wrong — and a wrong reason is worse than
a thin one, because the next series inherits it as precedent. The rule:
**an exclusion's justification is a claim, held to the same evidence
standard as a test-passing claim; assert the narrow condition only if you
verified it name by name, otherwise cite the broad rule that governs.**

### 2.4 §7 — the isolation-disposition rule (new, with both incidents)

Written into §7's own numbered procedure, immediately after step 4:
**"passes in isolation" is not a disposition unless it is n>1 AND paired
against the parent.** Two incidents, pointing opposite ways:

* wave-6 close — a name passed its isolated BASELINE 6 of 6, which reads
  like a branch-only regression; a matched batch of 10 per tree showed
  branch 5/10 red, baseline 4/10 red.
* wave-7 task 3 — a branch-unique name dispositioned "both PASS in
  isolation (`2 passed`)" off a SINGLE two-node selection. Re-run singly it
  is **5 failed / 5** on the branch and **2 failed / 3** at the isolated
  parent. The original claim asserted the OPPOSITE of the truth; the correct
  verdict survived only because someone re-measured.

Concrete four-step form: run the node ALONE; n≥3 (10 if timing/geometry
shaped); run the SAME batch at the parent in the isolated worktree; report
BOTH rates.

### 2.5 §7 — documented reds

Four new entries plus one re-confirmation, each with parent-proof
provenance:

| Entry | Provenance |
|---|---|
| `test_library_media_return_settlement.py::test_viewer_return_waits_for_geometry_then_scrolls_before_row_focus` and `::test_authoritative_recompose_rearms_before_replacement_geometry` | task 1: `assert (0, 45) == (0, 42)` byte-identical; whole file 2 failed / 42 passed on BOTH the branch and an isolated worktree at `83e17323e`. Flagged as worth more than an ordinary flake entry because they sit INSIDE the moved cluster |
| `test_persona_visual_runtime_boundary.py::test_persona_visual_modules_stay_outside_other_runtime_and_ui_boundaries` | task 1: 1 failed / 2 passed at `83e17323e` in the isolated worktree, identical. Surfaced only because task 1 ran the whole `Tests/Architecture` root |
| `test_library_selection_updates.py::test_tier1_toggle_falls_back_to_recompose_on_query_one_failure` | **already documented** (Task 9, wave 1) — recorded as a re-confirmation, not a duplicate entry, with WHY it read as new: the file had never been in a wave battery until the wave-7 fix round added a media guard to it. Paired branch 1f/6p vs isolated parent 1f/5p, the delta being exactly the new passing guard |
| `test_screen_navigation.py::test_media_route_round_trips_to_the_library_media_row` | task 3: 5 failed / 5 runs alone on the branch, 2 failed / 3 at the isolated parent `78186d159`, same `assert 'Screen' == 'LibraryScreen'`. Owner task **TASK-31881** |
| `test_library_honesty_accessibility.py::test_row_toggle_patcher_rebuilds_marker_label_both_directions` | this close: `assert '○ Export' == '○ Export selected'`, byte-identical in isolation on both trees. Given its own entry because of what it BLOCKS — see §4. Owner task **TASK-31880** |

### 2.6 §3 isolated-baseline and §9 probe rules — confirmed as standing procedure

Both were checked, not assumed. §3's rule reads "**The rule, restated with
no exception:** any baseline comparison — a whole-root sweep, a nine-file
paired battery, or a single node id — runs against an isolated `git worktree
add`…", with the carve-out explicitly dropped and priced. §9 carries "**Run
it as a PAIR on the same machine, and run the pair TWICE with the tree order
swapped**" with the wave-6 warm-up incident as its evidence and the
load-independent-columns reading rule. Both already read as standing
procedure; no edits made. The matched-batch rule that §9's neighbour needed
is now §7's step-4 disposition rule (§2.4 above).

### 2.7 §8 / §22 — the media rows

- §8's media row gains the **44%** exclusion framing and, more usefully, the
  WHY: the cause is TEST DEBT, not entanglement — 89 of the 111 exclusions
  are unbound-fake-self (73) or instance-attribute monkeypatch (16), fixture
  shapes each removable by retargeting a test, against 4 module-globals and
  5 genuine identity/lifecycle couplings. The 7 zero-mover-caller names are
  spelled out as the future MOVE candidates, with the pointer to §22's
  DECLINED verdict.
- §22's per-task table and pin-trajectory paragraph carry the corrected
  controller birth value (4474) and the down-then-up screen chain.
- A new **"Wave-7 close"** subsection carries the per-commit trajectory
  table, the verification battery, the paired sweeps, the probe and the
  lessons.

---

## 3. Stale-doc sweep (deliverable C)

### 3.1 The census, run repo-wide, with a positive AND a negative control

Tokenize-based (COMMENT and STRING tokens only for `.py`, plain line scan
for `.md`/`.toml`/`.tsv`/`.txt`/`.tcss`/`.sh`) over the **82 deleted flat
field names AND the 22 pruned method names**, across `tldw_chatbook/`, all
of `Tests/`, `Docs/`, `backlog/`, `scripts/` and `Helper_Scripts/`. Re-run
after the interruption, so these are the AFTER-fix numbers with recorded
output on disk:

```
files scanned: 11343
raw prose hits: 269
screen-attribution-narrowed (+/-3 lines): 94
```

**Positive control** (recipe §21 lesson 6 — a census that returns zero must
first be shown capable of returning non-zero): 269 > 0 over 11,343 files,
and the skip-list is matched on path parts RELATIVE to the repo root, so
this worktree's own `.worktrees` path segment cannot reproduce the wave-6
failure mode where a "clean sweep" was actually a sweep over zero files.

**Negative control, and it reconciles exactly.** The three sites task 3
handed forward are asserted by node and are now **absent** from the raw hit
set (`False` for all three). The before/after delta is verified
mechanically rather than remembered: applying the same boundary pattern to
this close's own diff over `Tests/` + `tldw_chatbook/` counts **6
occurrences removed, 0 added, net −6** — the six being the one in
`test_library_shell.py`, two in `test_library_multiselect_media.py`, one in
`test_library_media_render_fixes.py` and two in
`library_media_controller.py`. So the pre-fix census was **275**, and every
one of the six deltas is an intended edit rather than collateral.

**The ±3-line screen-attribution narrowing returned 94 candidates, and it
did not surface a single one of the three hand-off items.** That is the
point of the hand-off: task 3 ruled that widening the filter by hand inside
a cleanup PR is how a prose sweep stops being reproducible, and forwarded
them instead. Measured here, the narrowing's miss rate on this wave's known
defects is **3 of 3** — which is the strongest available argument for the
hand-off being a real mechanism rather than a courtesy.

### 3.2 The verdicts

Live line numbers verified against the current tree before any edit (all
three of task 3's hand-off line numbers were still exact).

| File:line | Defect | Fix |
|---|---|---|
| `Tests/UI/test_library_shell.py:9138` | "Selecting a different media row updates ``_selected_media_id``" | → ``_media_state.selected_media_id`` |
| `Tests/UI/test_library_multiselect_media.py:2153-2154` | the ADR-055 section comment names ``_library_media_bulk_delete_in_flight`` / ``_library_media_delete_receipt_ids`` as the live shared seam | → the dotted spellings the retargeted guard now asserts |
| `Tests/UI/test_library_media_render_fixes.py:1149` | quotes an ASSIGNMENT that exists nowhere: "set ``_library_media_view = "list"``" | → ``_media_state.view = "list"`` |
| `tldw_chatbook/UI/Library_Modules/library_media_state.py:389-392` | **a SECOND copy of the `_ANALYZE_ORIGIN_MEDIA` claim task 3 fixed in the module docstring** — the field comment still said it "stays a `library_screen` module constant" | → task 2 relocated the DEFINITION to `screen_constants.py`; `library_screen` imports it back, so the module-path attribute still resolves |
| `library_media_controller.py:160,165,167-168` | quotes the ADR-055 guard's PRE-retarget literal (`"_library_media_bulk_delete_in_flight = True"`) and two stale line refs (`library_screen.py:15501`/`:15505`) | → `self._media_state.bulk_delete_in_flight = True`, `:15062`/`:15066`; `test_review_set_walker.py:1284` → `:1292`. Line-neutral |
| `library_media_controller.py:553-555` (class docstring) | "keeps a one-line delegator for every one of the 140 original names … (this series' own cleanup PR … prunes …)" — present-tense and now FALSE | → "keeps a one-line delegator for **118** of the 140 … task 3 pruned the 22". Line-neutral |
| `library_media_controller.py:449-450` | "(deleted at cleanup, task 3, once this controller's own copy makes the screen's copy dead)" — reads as a plan | past-tensed. Line-neutral |
| `Tests/UI/test_library_media_characterization.py:16` | "~20 dedicated `Tests/UI/test_library_media_*.py` files" — task 1's review erratum M-6 said 16 and it was never applied to the tracked file | → 16 (live count: `ls` returns exactly 16) |
| `.git-blame-ignore-revs` (task-2 comment) | "the **91**-dependency construction site" — a FOURTH copy of a number whose other three copies task 2's erratum already fixed (83 construction site / 92 property surface) | → "1 positional + 83 keyword-only named dependencies; 92 is the hand-written property SURFACE, a deliberately different number". Re-derived by `ast` at this tree: 1 positional, 83 keywords, 92 `@property` |
| `.git-blame-ignore-revs` (task-3 comment) | "1,182 … across 32 files in FOUR Tests/ roots (UI 32, Architecture 2, Live 1, Media 1)" — conflates the 32 code-retarget files with the 36 CHANGED files whose roots the parenthetical actually splits | → "across 32 files — 36 Tests/ files changed in all, across FOUR roots (…)". Verified by `git show --name-only 5dd2e71cf -- Tests/`: 36 files, UI 32 / Architecture 2 / Live 1 / Media 1 |

`.git-blame-ignore-revs`' hash lines were not touched; only the prose above
them, which is not hash-load-bearing (the wave-6 close set that precedent).

### 3.3 What was deliberately NOT changed, and the line that decides it

The census's remaining live-file hits are ~12 prose sites in media test
files and controller docstrings that MENTION a flat name
(`_library_media_view`, `_library_media_detail`,
`_library_media_find_focus_pending`, …) without quoting an assignment or
attributing it to the screen. **They are not defects, and the reason is
mechanical rather than stylistic: every one of those flat names still
RESOLVES — permanently — on `LibraryMediaController`, whose own generated
shim loop is the copy the byte-for-byte canon requires.** Verified directly
(`hasattr(LibraryMediaController, "_library_media_view")` → `True`, and the
same for every name checked).

The operative line, stated so a reviewer can disagree with it precisely:
**a quoted assignment, a stale line reference, or an attribution to
`LibraryScreen` is a defect; a bare name-mention of an attribute that still
resolves on the controller is not.** All three hand-off items are on the
defect side of it (two quote a now-false literal, one attributes to a
real-screen Pilot fixture); the ~12 left alone are on the other. Frozen
historical records (`Docs/superpowers/plans|specs/*.md`,
`backlog/tasks/*.md`) were excluded from the candidate set by the standing
precedent task 3 recorded.

### 3.4 Relocated-constant mentions

Swept by name (`_ANALYZE_ORIGIN_MEDIA`, `_ANALYZE_ORIGIN_IMPORT`,
`_MEDIA_VIEW_LIST`, `_MEDIA_VIEW_VIEWER`) across `tldw_chatbook/`, `Tests/`,
`backlog/docs/` and `scripts/`/`Helper_Scripts/`. One stale copy found and
fixed (`library_media_state.py:391`, table above). Everything else is
accurate: `screen_constants.py:399-414` holds the definitions with a comment
saying `library_screen` imports them back; `library_media_controller.py:422-432`
states the relocation and names the two pinning assertions, whose live line
numbers (`test_library_ingest_analyze_skipped.py:611`/`:612`) were
re-verified exact; `test_screen_size_ratchet.py:625-626` is past-tense
accounting.

---

## 4. Follow-up filings (deliverable B)

**The id sweep found a real collision, exactly the one
`backlog/docs/lessons-backlog-hygiene.md` warns about.** Sweeping every
remote ref (`git for-each-ref refs/remotes/` + `git ls-tree -r <ref>
backlog/`, `sort -n`) plus the local tree gives a true maximum of **31860**.
The CLI auto-assigned **31821** to the first filing — an id already taken
**twice** on remotes
(`task-31821 - Close-remaining-inventory-UI-fixture-owned-database-resources`
and `task-31821 - Route-auth-account-login-bearer-writes-through-the-per-
profile-credential-scope`). Renumbered to **31880** (MAX+20) before anything
referenced the old id, with an `## Id provenance` section in the task file
recording it.

Neither red had an existing owner task: a content grep across
`backlog/tasks`, `backlog/drafts` and `backlog/archive` returned three files
naming `test_library_honesty_accessibility.py`, all **Done** and none owning
this failure, and zero naming either test node. Neither was §7-documented
before this close.

- **TASK-31880** — the honesty-accessibility row-toggle patcher red. Quotes
  the failing assert (`assert '○ Export' == '○ Export selected'`), records
  the both-trees isolation evidence, and states the coverage it blocks. AC
  includes "the fix does not make the test vacuous" and "remove recipe §7's
  entry once green".
- **TASK-31881** — the `test_media_route_round_trips_to_the_library_media_
  row` route-landing race. Quotes both measured rates (branch 5/5 red,
  parent 2/3 red) and names the sibling flaky route tests the recipe already
  documents, so the fix is asked to re-measure them too.

`scripts/check_backlog_task_ids.py`: **no duplicate task IDs across 3,367
task files** after both filings.

---

## 5. Verification battery (deliverable D.1)

**Every number below is from the POST-INTERRUPTION re-run** (see §8) — the
pre-interruption battery's output file was destroyed with the scratchpad and
counts as never-run. Load at battery time: **2.70 / 2.57 / 2.49** — a quiet
machine (the host had just rebooted).

| Suite | Result |
|---|---|
| Fresh `_measure()`, both ratchet files | 34669/1260 (screen), 4496 (controller) — exact match to both pins |
| **8 wiring suites + support-layer surface + both size ratchets + recompose ratchet + preimport closure + `_ui_ready` census + media characterization + screen-reuse + selection-updates (the new canvas-sync guard's home) + modal-dismissal**, ONE invocation | **301 passed / 5 failed**, 127.65s |
| ADR-055 interlock + the canvas-sync dotted-path guard + characterization + `test_library_media_render_fixes.py` | see §9 |
| `./scripts/preflight.sh` | see §9 |

**All 5 reds are on §7's documented list as updated by this close**:
`test_screen_does_not_grow_past_its_budget[chat_screen.py]`,
`test_task_22507_4_does_not_worsen_chat_screen_base`,
`test_controller_does_not_grow_past_its_budget[library_media_browse_
controller.py]` (410 vs 371 — still standing, still refused, a third
consecutive wave),
`test_tier1_toggle_falls_back_to_recompose_on_query_one_failure`, and
`test_library_modal_inventory_matches_declared_edges_bidirectionally` (the
skills-era discovery blocker, TASK-31815). Zero `library_screen.py`- or
`library_media_controller.py`-scoped failures. `_ui_ready` passed on this
run — it is a flapper and §7 says so.

---

## 6. The sweep found something no census could: a media state read reaching a SKILLS fixture

`Tests/UI/test_library_skills_canvas.py::test_action_library_skill_back_
honors_dirty_guard` failed on the branch with

    AttributeError: 'types.SimpleNamespace' object has no attribute '_media_state'

raised from `library_screen.py:10351`, inside
`_arm_library_list_entry_focus` — a SHARED shell seam, not a media method —
whose first statement task 3 retargeted to
`self._media_state.successful_focus_ownership = None`.

**Why every census in the recipe misses it.** The fixture is a
`SimpleNamespace` in a SKILLS file. It names **zero** media fields, so the
kwarg census cannot see it; it makes **zero** post-construction flat-name
assignments, so task 3's own "every receiver with a Store-context flat-name
attribute assignment" census — the one added precisely for the
`_entry_fake` case — cannot see it either. The flat name never appears in
the test at all. **The PRODUCTION method does the write, and the fake is
merely handed to it unbound.** Nothing static reaches that; only running it
does. (The fixture's own comment already records this exact recurrence for
a different attribute: "every `__init__`-only attribute the guarded-exit
path touches must be supplied here explicitly — this one was missing".)

**Disposition, measured on both trees rather than argued.** The test is a
PRE-EXISTING RED, not a wave regression:

| tree | result |
|---|---|
| branch, before this close's fix | `AttributeError: … no attribute '_media_state'` at `library_screen.py:10351` |
| isolated baseline `83e17323e` | `AttributeError: … no attribute 'focused'` at `library_screen.py:10742` |

Same test, same root cause (a stale unbound `SimpleNamespace` fake missing
attributes the seam reaches), different attribute — because at the wave
start the same write landed as a plain attribute CREATION on the namespace
and passed silently, so the fake got two lines further before tripping on
`self.focused`, which it never had.

**Fixed anyway, with the recipe's own prescribed one-line accommodation**
(§3's seventh bypass shape): `_media_state=SimpleNamespace(successful_focus_
ownership=None),` added to the fake, zero assertions touched. The test still
fails — on `focused`, which is a genuine dev-side gap outside this close's
remit — but it now fails **byte-identically on both trees**, so the wave
leaves no delta in a shared red and whoever fixes `focused` will not have to
re-derive the media one behind it. Verified after the edit: branch and
baseline both raise `AttributeError: 'types.SimpleNamespace' object has no
attribute 'focused'`.

**This is the generalizable finding of the close** and it is folded into the
recipe: the "bypassed-construction fake needs the nested state object to
EXIST" hazard is NOT confined to the subsystem's own test files. It reaches
any fake handed to a SHARED screen seam that happens to read one moved
field, in a file belonging to a different subsystem entirely.

---

## 7. A methodological failure of this close's own, found and repaired: the paired sweep was not paired

**The first attempt at both paired sweeps was VOID and was discarded before
any conclusion was drawn from it.** §3 mandates an isolated worktree with its
own fresh `uv venv` / `uv pip install -e ".[dev]"`. It does not say which
interpreter, and this task created the baseline with `uv venv --python 3.11`
— the project floor, and the version CLAUDE.md's own setup snippet suggests —
while the branch worktree's long-lived venv runs **Python 3.14.2**. Both
halves ran to completion and looked plausible.

**What gave it away was a legible tell, and it very nearly was not one.** The
`Tests/Library` + `Tests/Architecture` sweep's baseline-unique failures were
all parametrizations of `Tests/Architecture/test_python_floor_syntax.py::
test_detector_matches_the_pinned_floor_verdicts` — an f-string-nesting
detector whose cases parse differently before and after PEP 701. Any
Library-shaped failure produced by the same mismatch would have carried no
such signature and would have cost a full triage round, or been
mis-dispositioned as a regression.

**A second environment difference, found by the same check.** With the
interpreters matched, `uv pip list` still showed the fresh baseline as a
package SUPERSET of the branch's long-lived venv — the extras being
`html5lib`, `tinycss2` and `webencodings`. Its observable was a collection
`ModuleNotFoundError` on the BRANCH only, in
`Tests/UI/test_console_message_controller.py` — not a Library file,
byte-identical across the wave (`git diff 83e17323e HEAD --` on it is empty),
and none of its 17 tests selected by `-k "library"`. Repaired by re-running
`uv pip install -e ".[dev]"` in the branch worktree; that resync also moved
the branch venv's own editable `tldw-chatbook` metadata from `0.1.8.0` to
`0.2.0`, i.e. the venv had been stale for some time.

**Parity re-proven after the interruption, with recorded output** (§8 wiped
the first repair's evidence along with everything else):

```
python -V                 branch 3.14.2   baseline 3.14.2
uv pip list | awk .. sort IDENTICAL PACKAGE SETS (106 each)
tldw_chatbook.__file__    baseline resolves its OWN scratch tree
```

**Written into recipe §3** beside the isolated-baseline rule: *before reading
a paired sweep's counts, prove the two environments match — `python -V` on
both, and `uv pip list | awk '{print $1}' | sort` diffed between them.* Three
commands. It belongs beside §7's existing "reconcile the collection-count
asymmetry before reading the failure diff" step for exactly the same reason:
an unexplained baseline-unique failure is worth an hour, and an environment
difference will happily supply one.



---

## 8. The host interruption, and what was re-derived after it

**This task's host process exited mid-close, and the machine rebooted.**
`/private/tmp` is cleared on reboot, so the ENTIRE scratchpad went with it:
the isolated baseline worktree and its venv, every sweep output file, the
battery output, the census JSON, and the probe scaffolding. Recorded here
rather than smoothed over, because what survives an interruption and what
does not is the whole point of the recipe's evidence rules.

State at resume, characterized MECHANICALLY before any further edit:

| Check | Result |
|---|---|
| commits on top of `014f695fc` | **0** — everything uncommitted |
| dirty tracked files | 9 (mine) + `progress.md` (the coordinator's, untouched by me) |
| `git worktree list` | the scratch baseline listed as **prunable** — its directory was gone |
| every dirty `.py` compiles | **yes** (`py_compile`, all 7) |
| half-applied edit check: old spellings at every edited site (`15501`, `15505`, `:1284`, `91-dependency`, "stays a `library_screen` module") | **zero hits** — every edit fully applied |
| `_measure()` vs both pins | **34669/1260 == 34669/1260**, **4496 == 4496** |
| the two follow-up task files | present, untracked, ids intact |

**What counts as surviving.** The recipe's rule is that a run with no
recorded output is a run that did not happen. Applied literally here:

* **Re-derivable from the repo, so kept**: the per-commit pin trajectory
  (`git show` at 22 refs), the prose census (re-runnable), every line-number
  and count verification, the id sweep, both follow-up filings, and the tree
  edits themselves.
* **Destroyed, so re-run from scratch**: the isolated baseline worktree (+
  its venv), the guard battery, both paired sweeps (all four halves), and the
  probe. Every number in §5, §9 and §10 is from a post-reboot run.
* **Destroyed and NOT re-created**: the first, VOID py3.11 sweep pair (§7).
  Its files are gone and none of its counts is cited as evidence for
  anything; what survives from it is the METHOD finding, which is
  re-demonstrated above by a fresh parity check with recorded output.

**The interruption cost about 70 minutes of machine time and zero
conclusions**, because nothing had been written down that depended on a run
whose output was not on disk at the time of writing.

---

## 9. Paired baseline sweeps (deliverable D.2)

Isolated worktree at the wave-7 start `83e17323e`, own `uv venv` (3.14.2),
verified resolving its OWN tree. Run SEQUENTIALLY. Environment parity proven
BEFORE any count was read (§7).

| sweep half | started | load (1/5/15) | wall | failed | passed |
|---|---|---|---|---|---|
| A (`Tests/UI -k library`) — branch | 12:23 | 2.10 / 2.31 / 2.39 | 23:04 | 370 | 4129 |
| A — baseline | 12:47 | 2.91 / 3.85 / 4.36 | 24:05 | 369 | 4126 (+1 teardown error) |
| B (`Tests/Library` + `Tests/Architecture`) — branch | 13:11 | 3.61 / 3.98 / 4.82 | 3:10 | 23 | 3669 (5 skipped) |
| B — baseline | 13:14 | 6.19 / 5.79 / 5.46 | 3:14 | 23 | 3655 (5 skipped) |

Both halves of each sweep ran under comparable load and within 4% wall clock.

**Collection reconciliation, done before reading the failure diff:** Sweep A
branch 4,499 vs baseline 4,495 (**+4** = the 3 characterization tests + the
1 canvas-sync guard the wave added); Sweep B branch 3,697 vs baseline 3,683
(**+14** = 12 media-wiring tests + the 2 parametrizations the controller
ratchet's two `sorted(_BUDGETS)` decorators gain from the new row — verified
by counting the decorators, not assumed).

**Sweep B: 23 shared, ZERO branch-unique, ZERO baseline-unique.**

**Sweep A: 365 shared, 5 branch-unique, 4 baseline-unique.** Baseline-unique
not investigated per this section's established precedent. All 5
branch-unique dispositioned ALONE, n>1, PAIRED:

| name | branch | baseline `83e17323e` | verdict |
|---|---|---|---|
| `test_a_new_document_rescans_for_the_same_query` | 1/3 | 2/3 | documented (wave-5 task 2); worse on the tree without the wave |
| `test_loading_banner_paints_in_place_without_body_rebuild` | 1/3 | 2/3 | documented (wave-5 task 2); worse on the tree without the wave |
| `test_library_media_durable_mutation_gates_and_refreshes_applied_scope[True]` | 0/3 | 0/3 | documented (wave-5 task 2); passes alone on both |
| `test_home_origin_does_not_leak_into_later_library_entry` | **10/10** | **10/10** | NEW, deterministic on both — added to §7 |
| `test_library_skills_manual_items_priority_survives_compact_layout_sync` | see below | see below | NEW, load-sensitive geometry flapper |

**The fifth name is this close's most useful methodological result.** A
SEQUENTIAL matched batch gave branch **5/10** vs baseline **2/10** — an
asymmetry pointing the wrong way, in a file this close had itself edited,
i.e. the worst-looking outcome available. Re-run as **10 rounds ROUND-ROBIN
across three trees inside one load window**:

| tree | rate |
|---|---|
| branch (wave tip + this close's edits) | **2/10 failed** |
| wave tip `014f695fc` (no close edits) | **1/10 failed** |
| wave start `83e17323e` (no wave at all) | **5/10 failed** |

The direction fully reversed; the baseline became the worst tree. Nothing
about the code changed between the two experiments — only the interleaving.
The failure is a Skills adaptive-reader-shell geometry assertion
(`AssertionError: assert False … AdaptiveReaderEffectiveLayout(…,
items_open=False, …).items_open`), in a pane this wave's diff does not
touch. The third tree is what cleared this close's own edits specifically.

**Zero real regressions across the whole wave-7 span.**

---

## 10. Probe (deliverable D.3) — order-swapped pair

Quiet machine (load 1.88–2.10 throughout).
`Helper_Scripts/library_click_probe.py` is byte-identical across the wave
(`git diff 83e17323e HEAD --` on it is empty). Full tables are in the
recipe's wave-7 close; the verdict:

* **`recompose` is 0 on every row of all four runs.**
* **`full-update` follows the identical `2/2/2/1/1/1/1/1` pattern on both
  trees**, matching every close back to wave-2.
* **`nodes` is identical row-for-row between the trees** (119 media / 114
  notes).
* Branch wall-clock sits INSIDE wave-2 close's recorded band on every row:
  settle **244–464** against 264–485, max gap **41–129** against 54–195 —
  faster than the band at the low end of both.
* **No directional shift in either round, and none between them.** Wave-6's
  round 1 had shown the branch slower on all sixteen wall-clock cells from a
  first-run warm-up artifact; this close's probe ran immediately after ~90
  single-node pytest invocations, so both trees were already warm. The order
  swap stays mandatory precisely because you cannot know in advance which
  situation you are in.

---

## 11. Commits

| Commit | Kind | Subject |
|---|---|---|
| `f3ca373cc` | docs/tests-only | `chore(backlog): file the two wave-7 close follow-ups` |
| `f9f81019c` | docs/tests-only | `docs(library): wave-7 stale-doc sweep, and one fixture seed the sweep found` |
| `b7a170602` | docs/tests-only | `docs(recipe): wave-7 close — trajectory, two new identity forms, three new rules` |
| *(this commit)* | docs/tests-only | `docs(sdd): wave-7 task-4 close — durable evidence` |

The fourth row carries no hash on purpose: a commit cannot contain its own
hash, and an earlier draft that wrote one in was invalidated by the amend
that added this very line. The first three are `rev-parse`-stable and were
read from `git log`, never transcribed.

Four commits, in this order, all **docs/tests-only — no
`.git-blame-ignore-revs` HASH entry is added by any of them** (that file's
only change here is a prose correction to two existing entries, per the rule
that a docs-only close gets no blame-ignore row):

1. the two follow-up filings (TASK-31880, TASK-31881);
2. the stale-doc sweep (5 prose sites + the `.git-blame-ignore-revs` prose
   corrections + the skills fixture seed);
3. the recipe wave-7 close (§3 Forms D/E + the exclusion-honesty rule + §3's
   env-parity rule, §7's disposition rule + 5 documented-red entries, §8's
   media row, §22's corrections and the whole "Wave-7 close" subsection);
4. this report + the four task reports/briefs, force-added.

---

## 12. Deviations, errata and process notes

- **The `test_library_skills_canvas.py` fixture seed is the one non-doc
  edit in this close.** It is a single kwarg on a `SimpleNamespace`, zero
  assertions touched, and it is the accommodation recipe §3's seventh
  bypass shape prescribes. It does NOT make the test pass — the test is a
  shared pre-existing red on a different missing attribute — but it removes
  a wave-caused difference in HOW that red fails. Three-way evidence in §6.
- **Nothing was pushed. `progress.md` was not touched by me** (it carries
  the coordinator's own resume note and is deliberately left out of every
  commit below).
- **No `Docs/User_Guide/` page was touched, deliberately.** This close
  changes prose, a recipe, two backlog filings and one test fixture; it
  renders nothing differently. Consistent with all seven prior closes.
- **A count I could not fully reconcile, recorded rather than papered
  over.** Task 1 measured the media field-name census at **1,214
  occurrences across 36 files**; re-derived at this close against the same
  wave-start tree with the final 82-name set and an
  `(?<![A-Za-z0-9_])…(?![A-Za-z0-9_])` boundary it measures **1,204 across
  34**. The ~10-occurrence gap is an instrument difference neither run can
  now reconstruct. Both figures are recorded in §22 rather than one
  overwriting the other — what IS newly established, and mechanically, is
  that the census's root split (UI 31 / Library 1 / Architecture 1 / Live
  1) is a DIFFERENT set from the changed-file split (UI 32 / Architecture 2
  / Live 1 / Media 1), and that the "five roots" the program has quoted is
  their union.
- **The isolated worktrees** (`w7close` @ `83e17323e` and `w7tip` @
  `014f695fc`, each with its own `uv venv`) are removed at task close.

---

## 13. Concerns handed forward

1. **`library_media_browse_controller.py`'s standing red survives a third
   wave** (410 vs 371). No wave-7 commit edited that file, so the
   absorb-only-if-touched rule left it flagged. It is dev-owned creep and
   the ratchet's own guidance forbids re-pinning it green from a passing
   branch.
2. **TASK-31880 blocks real coverage, not just a test.** Until it lands,
   the media row-toggle patcher path has no real-row Pilot coverage and the
   fifth-spelling fix rests on a screen double alone.
3. **Notes is next (§8), and it inherits three things media proved.** It
   WILL own `on_<message>` name-dispatched handlers (media owned zero, so
   §4's third whitelist member was inert this wave and is untested since
   prompts); it is the third subsystem to need the dotted branch in
   `canvas_sync.py`'s computed name, and should add its own three-line
   guard sibling; and its cleanup should expect the shared-shell-seam
   fixture hazard (§6) in files belonging to other subsystems.
4. **The 7 zero-mover-caller media methods remain move candidates** for a
   separately-motivated PR, re-evaluated and DECLINED at task 3 and not
   revisited here.
