# Wave-5 final-review fix report — dev reconciliation merge + conditions

Executes `final-fix-brief.md` against worktree
`/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation`,
branch `refactor/library-decomp-wave5-ingest`, starting at HEAD `e9eca2a38`.

**Commits created (this session, in order):**

| # | Hash | Subject |
|---|---|---|
| 1 | `897ab81f792852b0a90a1089b174de7f58a290e1` | `Merge origin/dev into refactor/library-decomp-wave5-ingest` |
| 2 | `3612e7c76a2b46158163b2b6683fdd39126a9cc9` | `docs(library): wave-5 final-review conditions — blame-ignore backfill, count corrections, follow-up filed` |

Nothing was pushed. No rebase, no squash. `progress.md` was never staged or
edited (it carried a pre-existing unstaged modification when this session
started and still does).

---

## 1. The merge

```
$ git fetch origin
$ git rev-parse origin/dev           -> 93388ba69b7499c2bc3180fc26c82d7f341871a7
$ git merge-base HEAD origin/dev     -> 68f9d865fad623db6ec02e19632090c1140b3c89
$ git rev-list --count 68f9d865f..origin/dev  -> 89
```

89 dev commits since the branch's merge-base — exactly the drift the
reviewer measured; dev did not move further between the review and this
merge.

`git merge origin/dev` produced **one** conflicted file, `tldw_chatbook/UI/
Screens/library_screen.py`, in **two** hunks — matching the brief. 186 files
changed in total (`17689 insertions(+), 3470 deletions(-)`).

**No diagnostic-inventory conflict.** `Docs/security/production-diagnostic-
inventory.json` and `scripts/check_persistent_diagnostic_inventory.py` both
auto-merged cleanly, and `./scripts/preflight.sh` reports the inventory
verified with no drift (§6.5 below). Nothing was regenerated; `--write` was
never run.

**No other conflict of any kind**, so no "resolve mechanically toward dev"
cases arose.

### 1a. Import hunk (`library_screen.py`, the `Library.library_ingest_state` block)

Conflict shape: HEAD's side was empty (the wave deleted both progress
helpers); dev's side added three names.

Resolution — dev's side taken, minus the two dead names:

| Name | Verdict | Evidence |
|---|---|---|
| `library_ingest_analyze_skipped_ids` | **KEEP** | dev's new `handle_library_ingest_analyze_skipped` calls it |
| `format_ingest_progress_line` | **DROP** | see grep below |
| `ingest_progress_action_signature` | **DROP** | see grep below |

Verification that the two drops are dead on the screen, run on the
post-resolution file:

```
$ grep -n 'format_ingest_progress_line\|ingest_progress_action_signature' \
       tldw_chatbook/UI/Screens/library_screen.py
(no output; exit 1)
```

And the reason they are dead — at the merge base both were used by exactly
one screen method, which task 2 moved to the controller:

```
$ git show 68f9d865f:tldw_chatbook/UI/Screens/library_screen.py \
    | grep -n 'format_ingest_progress_line\|ingest_progress_action_signature'
161:    format_ingest_progress_line,
162:    ingest_progress_action_signature,
20315:        if ingest_progress_action_signature(before) != ingest_progress_action_signature(
20329:        progress_widget.update(format_ingest_progress_line(progress, state=after.state))
```

Lines 20315/20329 are inside `_handle_library_ingest_progress_changed`, a
mover (it is in `_INGEST_CLUSTER_METHOD_NAMES`). **Dev added no new
screen-side use of either name** — hence the drop, not a keep.

Final block (post-merge):

```python
from ...Library.library_ingest_state import (
    validate_ingest_option_value,
    INGEST_UNAVAILABLE_COPY,
    LibraryIngestCanvasState,
    LibraryIngestLastSubmission,
    active_ingest_start_confirm_line,
    build_library_ingest_state,
    clamp_chunk_size,
    library_ingest_analyze_skipped_ids,
    library_ingest_retry_available,
    library_ingest_retry_label,
    parse_keywords,
)
```

### 1b. Method hunk (the ingest-handler region)

Conflict spanned HEAD lines 33837–34028: HEAD's side was 5 lines (two
delegators); dev's side was 184 lines (two full pre-move bodies, two new
methods, one class constant).

| Element | Side taken | Why |
|---|---|---|
| `_on_ingest_job_details` | **HEAD** (one-line delegator) | dev's side is the full pre-move body and calls `self._ingest_job_id_from_button`, which task 3 pruned from the screen. Taking dev's side reintroduces a call to a deleted name. |
| `handle_library_ingest_clear_finished` | **HEAD** (one-line delegator) | canonical body lives in the controller; dev's edit ported there instead (§2). |
| `handle_library_ingest_analyze_skipped` | **dev**, verbatim | dev's NEW handler; kept on the screen exactly as written. |
| `_record_library_ingest_analyze_outcome` | **dev**, verbatim | dev's NEW helper; kept on the screen exactly as written. |
| `_CLEAR_FINISHED_DEAD_ZONE_SECONDS = 0.3` (+ its 2 `#:` comment lines) | **dropped from the screen** | task 2 moved this class constant to the controller and deleted it from the screen; dev's side is just the pre-move text. See grep below. |

Byte-for-byte verification of the two kept dev methods and of the delegated
body, by AST body-range extraction against `origin/dev`:

```
handle_library_ingest_analyze_skipped   screen == dev screen : True
_record_library_ingest_analyze_outcome  screen == dev screen : True
_on_ingest_job_details                  controller == dev screen : True
handle_library_ingest_clear_finished    controller == dev screen : True   (see §2)
```

Dead-name census on the post-resolution screen (all four expected to be
absent):

```
$ grep -n 'format_ingest_progress_line\|ingest_progress_action_signature\|
           _ingest_job_id_from_button\|_CLEAR_FINISHED_DEAD_ZONE_SECONDS' \
       tldw_chatbook/UI/Screens/library_screen.py
(no output; exit 1)

$ grep -rn '_CLEAR_FINISHED_DEAD_ZONE_SECONDS' tldw_chatbook/ Tests/
tldw_chatbook/UI/Library_Modules/library_ingest_controller.py:372   (docstring)
tldw_chatbook/UI/Library_Modules/library_ingest_controller.py:472   (the constant)
tldw_chatbook/UI/Library_Modules/library_ingest_controller.py:2458  (its one reader)
```

Runtime confirmation on the merged tree:

```
screen has _CLEAR_FINISHED_DEAD_ZONE_SECONDS : False
screen has _ingest_job_id_from_button        : False
screen has handle_library_ingest_analyze_skipped   : True
screen has _record_library_ingest_analyze_outcome  : True
```

Every name dev's two kept handlers reach is present on the screen (a KEEP
delegator or a still-screen-resident excluded method):

```
_library_ingest_registry              True   (mover, KEEP delegator)
_start_library_media_analyze          True   (Media, never moved)
_update_library_ingest_dynamic_regions True  (excluded, screen-resident)
_build_library_ingest_state           True   (excluded, screen-resident)
```

---

## 2. Porting dev's clear-finished edit into the controller

Dev's post-move edit (task-28007's Qodo review round, PR #2400 #2) added a
13-line block to `handle_library_ingest_clear_finished` that pops
`self._library_ingest_analyze_outcomes` for each terminal job. That method's
canonical body is now `LibraryIngestController`'s, so **the edit follows the
body**: the block is ported into the controller and the screen keeps its
one-line delegator. This is the series' first deliberate divergence from the
"byte-for-byte with the pre-extraction body" canon, and it is recorded in
the merge-commit message, in the controller's module docstring (a new
paragraph, *One body follows a post-move dev edit (reconciliation merge)*),
and here.

### Accessor wiring diff summary

Mirrors the `library_canvas_resync_pending_accessor` precedent exactly
(`library_ingest_controller.py:502/587-589/715-721`), minus the setter.

| Where | Change |
|---|---|
| `LibraryIngestController.__init__` signature | new keyword-only param `library_ingest_analyze_outcomes_accessor`, in group (b) with a 4-line comment. Keyword-only arity **37 → 38** (`inspect.signature` total incl. `self` and `screen`: 39 → 40). ERRATUM (scoped re-review): this row originally said "38 → 39", a figure carried from the fix brief without measuring; the merge commit 897ab81f7's message carries the same stale figure and stays as-is (amending would rewrite the branch tip); the PR description carries the erratum. |
| `LibraryIngestController.__init__` body | `self._library_ingest_analyze_outcomes_accessor = (library_ingest_analyze_outcomes_accessor)` |
| `LibraryIngestController` properties | new `@property _library_ingest_analyze_outcomes` returning `self._library_ingest_analyze_outcomes_accessor()`. **Getter only, deliberately**: the ported block only `pop`s, and a `dict` mutates in place through the getter — which is exactly what keeps the ported lines byte-for-byte with dev's. |
| `handle_library_ingest_clear_finished` body | dev's 13 lines (10 comment + 3 code) inserted verbatim, between the `terminal = [...]` comprehension and `known = {...}`, at dev's own position. |
| module docstring | group (b) count 7 → 8 with the new name; new divergence paragraph naming task-31651 as the interim bridge's exit. |

**Verbatim proof** — AST body-range extraction, `origin/dev`'s
`LibraryScreen.handle_library_ingest_clear_finished` vs the merged tree's
`LibraryIngestController.handle_library_ingest_clear_finished`:

```
IDENTICAL: True
```

### Construction sites updated (all 3)

`grep -rn 'LibraryIngestController(' tldw_chatbook/ Tests/` finds exactly
three:

| Site | Accessor passed |
|---|---|
| `tldw_chatbook/UI/Screens/library_screen.py:2523` (production) | `lambda: self._library_ingest_analyze_outcomes` |
| `Tests/UI/test_library_shell.py` `wire_bypass_ingest_controller` | `lambda: screen._library_ingest_analyze_outcomes` |
| `Tests/UI/test_library_ingest_inline_consent.py` `_wire_bypass_ingest_controller` | `lambda: screen._library_ingest_analyze_outcomes` |

Both bypass helpers additionally seed the flat field
(`if not hasattr(screen, "_library_ingest_analyze_outcomes"): screen.
_library_ingest_analyze_outcomes = {}`) for the same reason they already
seed `_ingest_controller`: `object.__new__(LibraryScreen)` skips the
`__init__` line that creates it. This is recipe §3's "seventh bypass shape"
applied to dev's new field.

### Wiring-test arity pin

**None exists.** `Tests/Architecture/test_library_ingest_wiring.py` pins the
mover count (56), the pruned set, the staticmethod set, and the state-field
shim surface — but no constructor arity, and no repo-wide grep for `arity`
or `inspect.signature` in the library wiring tests finds one. Brief step 2.4
was therefore a no-op; nothing was changed there for arity.

---

## 3. Post-merge census re-verification

### 3a. Boundary-anchored 20-field census

Every `LibraryIngestState` field name, in its flat `._library_ingest_<field>`
form, word-boundary anchored, across all of `tldw_chatbook/**/*.py` and
`Tests/**/*.py`, excluding `library_ingest_controller.py` (which reaches
them through its generated shims) and `UI/Library_Modules/
library_ingest_state.py` (which defines them):

```
Tests/UI/test_library_ingest_analyze_skipped.py:647:
    screen._library_ingest_clear_finished_armed_at -= 1.0
tldw_chatbook/Library/library_ingest_state.py:1215:
    ``self._library_ingest_form`` before it) rather than a scatter of scalar
TOTAL: 2
```

**Hit 1 — a real new flat use, retargeted.** Dev's new test file reaches
the armed-at stamp as a flat screen attribute; the screen's generated shim
block was deleted wholesale in the ingest cleanup PR, so this is an
`AttributeError` waiting to fire (`-=` reads before it writes). Retargeted
with the same mechanical receiver swap task 3 applied 297 times, with a
one-line reason comment:

```python
        # (wave-5 merge) The armed-at stamp is a `LibraryIngestState` field
        # now, not a flat screen attribute -- the screen's generated shim
        # block was deleted in the ingest cleanup PR.
        screen._ingest_state.clear_finished_armed_at -= 1.0
```

**Hit 2 — not a use.** Prose inside a docstring in the unrelated pure-logic
module `tldw_chatbook/Library/library_ingest_state.py` (a different module
from `UI/Library_Modules/library_ingest_state.py`). Left alone: it is dev's
file, it is a historical narrative sentence, and it executes nothing.

Zero flat uses remain in `library_screen.py` itself — the reviewer's
"zero outside the conflict hunk" result reproduces at 89 commits of drift.

### 3b. Dev's task-28007 tests pass unmodified

- `Tests/UI/test_library_ingest_analyze_skipped.py` (dev's new 744-line file,
  the one that actually exercises the ported block) — **14 passed**,
  standalone, with **zero** modifications other than the one census retarget
  above. Its
  `test_clear_finished_prunes_the_stale_outcome_for_a_reused_media_id` drives
  the arm/confirm sequence through the screen's delegator into the
  controller's ported block, so it is direct proof of the port.
- `Tests/UI/test_library_ingest_canvas.py` — see §6.4 (2 failures, identical
  on `origin/dev`).
- `Tests/UI/test_library_shell.py` — see §6.4. Dev's changes to this file are
  Media-scoped (task-28007 AC#5 and task-31220), not ingest-handler tests.

---

## 4. Ratchet re-measurement (in the merge commit)

Both re-measured **after** the merge, with each ratchet file's own
`_measure()` semantics, not `wc -l` guesses.

| Row | Before | After |
|---|---|---|
| `test_screen_size_ratchet.py` — `LibraryScreen` | `(40094, 1296)` | **`(41028, 1313)`** |
| `test_library_modules_size_ratchet.py` — `library_ingest_controller.py` | `2569` | **`2623`** |

**Screen, +934 lines / +17 methods.** The method delta is the load-bearing
check: measured on the same class,

```
merge-base 68f9d865f : (41574, 1302)
origin/dev 93388ba69 : (42518, 1319)     -> dev's own delta: +944 / +17
branch     e9eca2a38 : (40094, 1296)
merged     897ab81f7 : (41028, 1313)     -> 1296 + 17 = 1313 exactly
```

An exact `+17` means the resolution re-added **no** moved body and lost
**no** wave-5 delegator — the two failure modes the reviewer proved a naive
resolution produces. Line composition: `+861` from dev's auto-merged hunks
elsewhere in the file, `+69` for dev's two new screen-resident ingest
methods, `+3` for the new accessor binding at the construction site, `+1`
for `library_ingest_analyze_skipped_ids` (the 2 dropped imports account for
the rest of dev's 3-line import addition). `861+69+3+1 = 934`.

**Controller, +54 lines (2569 → 2623)**, itemised in the pin's own comment:
`+13` ported block, `+5` constructor param and its comment, `+3` storage,
`+12` property, `+21` module docstring (group-(b) rewrite, the divergence
paragraph, task-31651). No other body touched; the 56 movers are unchanged.

Both files use the comment convention prior catch-up merges established in
`test_screen_size_ratchet.py` (the wave-3 `#:` re-measure note and the
wave-4 "final review: `origin/dev` merge (106 commits…)" entry): a dated
entry stating the commit count, the merge-base SHA, the fresh measure, and
why it rose.

### Ratchet results

`Tests/Architecture/test_screen_size_ratchet.py` +
`test_library_modules_size_ratchet.py`: **3 failed, 33 passed** — and all
three failures are red on `origin/dev` itself, verified by running both
files in an isolated `origin/dev` worktree with its own venv:

```
branch     : 3 failed, 33 passed
origin/dev : 4 failed, 30 passed   (the same 3, plus the library_screen row
                                    this merge just re-pinned)
```

| Row | Status |
|---|---|
| `chat_screen.py` budget | pre-existing dev red (documented, recipe §7) |
| `test_task_22507_4_does_not_worsen_chat_screen_base` | pre-existing dev red (documented, recipe §7) |
| `library_media_browse_controller.py` | **pre-existing dev red, NOT in the brief's expected list** — see §7 concerns |

---

## 5. Condition / minor fixes (commit 2, `3612e7c76`)

### M1 — `.git-blame-ignore-revs`, "3 sites" → "2 sites"

The wave-5 task-1 entry claimed the fixture sweep "still missed" 3 sites.
Verified true count:

```
$ git show 74a6f5774 -- Tests/UI/test_parakeet_v2_install_ui.py \
    | grep -c '^+.*_ingest_state = LibraryIngestState()'
2
$ git show --stat 74a6f5774 --format=""   # only ONE test file touched
 .../progress.md                              |   5 +
 .../task-1-report.md                         | 211 ++++--
 Tests/UI/test_parakeet_v2_install_ui.py       |   3 +
 backlog/docs/library-decomposition-recipe.md  |  75 ++
```

Corrected to "the 2 sites this sweep still missed (both in
`Tests/UI/test_parakeet_v2_install_ui.py`)" — the file is now named, so the
next reader does not have to re-derive it.

### Blame-ignore prior-wave gap (ruled FIX-FORWARD)

The same entry recorded that every wave-2/3/4 **state** PR was missing from
this file and left it unfixed. Fixed forward: four entries appended in their
own chronological slots (immediately before each wave's controller entry,
matching the conversations exemplar's own backfilled placement).

**Hash-verification transcript — every hash below came from `git rev-parse`
output in this session; none was typed from memory.**

```
$ git log --oneline --all --grep='series 1/3'
12ba4fb13 refactor(library): ingest state object + shims (ingest series 1/3)
a11220648 test(library): characterization + wiring pins for the ingest extraction series (RED)
87c318d57 refactor(library): skills state object + shims (skills series 1/3)
77750c85d refactor(library): search+RAG state object(s) + shims (series 1/3)
bca923b4c refactor(library): collections state object + shims (collections series 1/3)
2ccfccbc7 test(library): characterization pins for the collections extraction series
f4e8acecf refactor(library): export state object + shims (export series 1/3)

$ for h in f4e8acecf bca923b4c 77750c85d 87c318d57; do
      printf "%s -> %s | " "$h" "$(git rev-parse "$h")"; git log -1 --format="%s" "$h"; done
f4e8acecf -> f4e8acecf3366bdb83700d40d8fc2a3096ad0e0b | refactor(library): export state object + shims (export series 1/3)
bca923b4c -> bca923b4cea346ece8cabe94433a319b6b1566b1 | refactor(library): collections state object + shims (collections series 1/3)
77750c85d -> 77750c85dbd69f2ab4c25fefb8fcb8efc55d0621 | refactor(library): search+RAG state object(s) + shims (series 1/3)
87c318d57 -> 87c318d57cd91b786dd485f26fd103f8fca20628 | refactor(library): skills state object + shims (skills series 1/3)
```

The reviewer's `87c318d57` for wave-4 skills is **confirmed correct**.

Each was additionally checked to be an ancestor of HEAD and a genuine pure
state move (`git merge-base --is-ancestor` + `git show --stat`):

| Hash | Wave / task | Files touched | Ancestor of HEAD |
|---|---|---|---|
| `f4e8acecf…0e0b` | wave-2 task 2, export state, 13 fields | wiring test, ratchet, recipe, new state module, `library_screen.py` | yes |
| `bca923b4c…66b1` | wave-2 task 5, collections state, 26 fields | wiring test, ratchet, recipe, new state module, `library_screen.py` | yes |
| `77750c85d…0621` | wave-3 task 2, search+RAG state, 20 fields | ratchet, `library_screen.py` | yes |
| `87c318d57…0628` | wave-4 task 1, skills state, 36 fields | ratchet, `library_screen.py` | yes |

The `12ba4fb13` comment's closing sentence ("not fixed retroactively for
those here") is replaced by the fix-forward record naming all four.

### M2 — prune-fraction denominator (`test_library_ingest_wiring.py`)

The comment quoted **6-of-29 (~21%)** against a range whose every endpoint
uses a *total-mover* denominator. Corrected to **6-of-56 (~11%)**, the low
end of that range, with the range's own endpoints spelled out (export
1-of-22, skills 16-of-86, collections 14-of-64, search+RAG 12-of-42,
conversations 18-of-61) and the old figure named so it is not reintroduced.
Source: task-3 report §4, "Delegator census — 50 KEEP, 6 PRUNED", which
already states "**6 of 56 (~11%)** — the LOW end of every prior series'
recorded fraction".

### M3 — recipe §8 ingest row parenthetical

`(24-vs-27-site undercount from a -k-filtered sweep missing a file)`
conflated the review's CRITICAL with a separate Important. Reworded to §20's
own framing: the CRITICAL was **2 tests left RED at HEAD** in
`Tests/UI/test_parakeet_v2_install_ui.py` — the one file whose filename and
test names contain neither "ingest" nor "library", so every `-k`-filtered
sweep was structurally blind to it — a no-red-ships violation; the
24-vs-27-site count error in the same task's report was a **distinct**
Important finding.

### I2 follow-up task — **TASK-31651**

Filed as `backlog/tasks/task-31651 - Fold-the-ingest-bulk-Analyze-outcomes-
field-into-LibraryIngestState-and-census-devs-two-new-ingest-methods.md`,
with 8 individually-checkable acceptance criteria (repeated `--ac` flags;
`-l library,refactor` as a single comma list, per the two flags' inverted
comma semantics) covering: the field becoming a 21st `LibraryIngestState`
field, retargeting dev's screen reads, removing the interim accessor added
in this merge, censusing `handle_library_ingest_analyze_skipped` and
`_record_library_ingest_analyze_outcome` into the cluster, updating the
wiring test's pins and the recipe's §8/§20 ingest rows, re-pinning both
ratchets, and dev's task-28007 tests still passing.

**Id chosen after a true max-id sweep across refs, not the local tree.**

```
$ git branch -a --format='%(refname:short)' | wc -l
582
$ (for each of the 582 refs) git ls-tree -r --name-only <ref> \
      -- backlog/tasks backlog/drafts backlog/archive \
   | grep -oE 'task-[0-9]+' | sed 's/task-//' | sort -n -u
...
31645
31650          <- global max across all local + origin refs (3012 distinct ids)
```

`31651` was free everywhere. `preflight.sh`'s duplicate-id check
subsequently passed across 3,282 task files.

---

## 6. Verification battery

All from the worktree with `.venv/bin/python -m pytest`, `-p no:randomly`.

### 6.1 + 6.2 + 6.3 — wiring suites, characterization files, both ratchets, recompose census guard, support-layer surface

Run together (6 wiring suites + 4 ingest/collections/conversations/export
characterization files + both size ratchets + `test_library_recompose_
ratchet.py` + `test_library_support_layer_surface.py`):

```
3 failed, 104 passed in 73.42s
```

The 3 failures are the pre-existing `origin/dev` ratchet rows itemised in
§4. Every library wiring suite, every characterization file, the recompose
census guard and the support-layer surface test are green.

### 6.3b — inline-consent + ingest canvas + dev's new analyze-skipped file

```
Tests/UI/test_library_ingest_inline_consent.py
Tests/UI/test_library_ingest_analyze_skipped.py
Tests/UI/test_library_ingest_canvas.py
  -> 2 failed, 205 passed in 69.70s
```

Both failures are
`test_library_ingest_canvas.py::test_progress_detail_paints_below_row_without_obscuring_actions_or_neighbor[size0|size1]`,
and they are **identical on `origin/dev`**:

```
origin/dev worktree, same file, same invocation:
  FAILED ...test_progress_detail_paints_below_row_without_obscuring_actions_or_neighbor[size0]
  FAILED ...test_progress_detail_paints_below_row_without_obscuring_actions_or_neighbor[size1]
  2 failed, 136 passed in 42.84s
```

They also appear in **both** sides of the xdist sweep, so they are in the
shared backdrop, not a branch-unique name.

Standalone: `Tests/UI/test_library_ingest_analyze_skipped.py` — **14 passed**.

### 6.4 — dev's task-28007 test files, run as files

Dev's edits to these two files are Media-scoped (task-28007 AC#5's
disabled-Generate assertions and task-31220's row/undo gate changes in
`test_library_shell.py`; a 4-line change in `test_library_ingest_canvas.py`).
Dev's actual task-28007 *ingest* coverage is the new file
`Tests/UI/test_library_ingest_analyze_skipped.py`, run standalone above.
Both named files were run as whole files, single-process, on the branch AND
on the `origin/dev` worktree with the identical invocation:

| File | branch | `origin/dev` | branch-unique |
|---|---|---|---|
| `Tests/UI/test_library_ingest_canvas.py` | 2 failed, 136 passed | 2 failed, 136 passed | **0** (same two names) |
| `Tests/UI/test_library_shell.py` | 227 failed, 598 passed (1:26:36) | 226 failed, 599 passed (1:26:10) | **1** |

`test_library_shell.py`'s 226 shared failures are the file's own documented
standalone-run collapse (task 3's `fd_leak_sentinel` lead: a 274-fd growth
on a standalone run of this file, with the `test_library_note_*` DOM-timeout
burst both trees develop identically near the tail). Identical on both
sides, so not attributable to this merge.

The single branch-unique name,
`test_library_shell.py::test_library_media_initial_error_is_unknown_and_retry_is_unique`
(Media, not ingest), passes in isolation on **both** trees:

```
branch     : 1 passed in 6.89s
origin/dev : 1 passed in 5.95s
```

— ordering noise inside a file where 226 other tests are already failing
identically on both sides. **Zero branch-unique real failures.**

### 6.5 — `./scripts/preflight.sh`

**All green**, six checks:

```
=== generated stylesheets ===        all 10 bundles reproduce
=== profile-owned path census ===    48 occurrences / 18 files, all matched
=== production diagnostic inventory ===
                                     574 owners, 1336 TASK-492 calls,
                                     7603 TASK-494 calls, 10 sink files
=== backlog task ids ===             no duplicates across 3282 task files
=== chachanotes table allowlist ===  105 declared tables, all present
=== index plan pins ===              270 indexes / 270 census rows / 57 pinned
preflight: all derived-artifact checks passed.
```

No drift was reported, so `check_persistent_diagnostic_inventory.py --write`
was never run.

### 6.6 — Paired-baseline sweep (recipe §7)

Isolated `git worktree --detach` at `origin/dev` (`93388ba69`) placed under
the session scratchpad — **never** an in-place checkout in this worktree —
with its **own** `uv venv` and `uv pip install -e ".[dev]"` (Python 3.14.2
on both sides). Import provenance verified before trusting a single result:

```
$ cd <devbase> && ./.venv/bin/python -c "import tldw_chatbook; print(tldw_chatbook.__file__)"
<scratchpad>/devbase/tldw_chatbook/__init__.py
```

Sweeps run **sequentially** (never two 8-worker invocations at once), same
invocation on both sides:

```
.venv/bin/python -m pytest Tests/UI -k "library" -p no:randomly -q -n 8 --dist worksteal
```

| Side | Result | Wall |
|---|---|---|
| branch (merged tree) | **356 failed, 4074 passed** | 22:31 |
| `origin/dev` baseline | **354 failed, 4067 passed** | 23:20 |

Failure-name diff: **351 shared**, **5 branch-unique**, **3 baseline-unique**.

Branch-unique names, each resolved per §7 step 4 — first combined
single-process, then in isolation, then the identical isolation run on the
`origin/dev` worktree:

| Branch-unique name | branch, combined | branch, isolated | **origin/dev, isolated** |
|---|---|---|---|
| `test_library_media_reader_match_nav_t22209.py::test_a_new_document_rescans_for_the_same_query` | pass | pass | pass |
| `test_library_media_reader_traversal_t22207.py::test_one_megabyte_markdown_document_is_not_reparsed_per_keystroke` | pass | fail | **fail** |
| `test_library_notes_reader.py::test_wide_editor_deep_link_keeps_reader_navigation_and_local_back` | pass | fail | **fail** |
| `test_library_prompts_canvas.py::test_library_prompt_undo_refreshes_applied_page_and_preserves_basket` | pass | pass | pass |
| `test_library_shell.py::test_library_shell_note_id_deeplink_opens_note_editor` | pass | fail | **fail** |

All five behave **identically on `origin/dev`** under the same
single-process invocations — the three that fail in isolation fail on dev
too, the two that pass in isolation pass on dev too. The combined
single-process run of all five is `5 passed` on the branch and
`1 failed, 4 passed` on dev, flipping in both directions between runs: the
signature of §7's documented xdist/ordering noise, in a Notes/Media/Prompts
neighbourhood this merge does not touch.

**Zero branch-unique real failures.** None of the 5 is ingest-scoped; none
reproduces as a branch-only defect.

The throwaway `origin/dev` worktree was removed afterwards
(`git worktree remove --force` + `git worktree prune`); `git worktree list`
shows only the pre-existing worktrees.

---

## 7. Concerns / notes for the reviewer

1. **A THIRD pre-existing `origin/dev` ratchet red, not named in the brief.**
   `Tests/Architecture/test_library_modules_size_ratchet.py::test_controller_
   does_not_grow_past_its_budget[tldw_chatbook/UI/Library_Modules/library_
   media_browse_controller.py]` — the file is 410 lines against a 371 pin.
   Both the file and the pin come wholly from `origin/dev` (`git show
   origin/dev:<file> | wc -l` → 410; `git show origin/dev:<ratchet>` → 371),
   and running both ratchet files in the isolated `origin/dev` worktree
   reproduces it there. This branch never touched that controller. **I did
   not raise the pin**: absorbing dev's un-owned creep into this merge would
   silence a guard that is correctly firing at dev, and it is not this
   wave's debt. It needs an owner on dev (same category as the two
   `chat_screen.py` rows already documented in recipe §7 and
   backlog task-31249's "Library UI test debt on dev").

2. **The interim accessor is a knowingly incomplete boundary.**
   `_library_ingest_analyze_outcomes` is now the only ingest field the
   controller reaches through a bespoke binding rather than through the
   state object — the ingest subsystem's state boundary is 20 fields where
   it should be 21. That is deliberate (the brief's chosen shape, and the
   minimum-risk way to keep dev's edit byte-for-byte in a merge commit),
   time-boxed by **task-31651**, and cross-referenced from the controller's
   module docstring so it cannot quietly become permanent. Until it lands,
   the ingest cluster's census is also two methods short: dev's
   `handle_library_ingest_analyze_skipped` and
   `_record_library_ingest_analyze_outcome` are ingest-named screen methods
   that the wave never censused (they did not exist at the wave's base).

3. **`Tests/UI/test_library_ingest_canvas.py`'s 2 failures are real
   failures that ship on dev.** They are identical on `origin/dev` and on
   both sides of the xdist sweep, so nothing in this merge caused them and
   nothing here can be blamed for them — but they are *ingest*-scoped and
   red, so the ingest subsystem does not have a fully green file. Flagging
   rather than fixing: out of this brief's scope, and dev owns them.

4. **`test_library_shell.py` standalone is 226/825 red on BOTH trees.** Task
   3 already recorded the lead (`fd_leak_sentinel`, a 274-fd growth on a
   standalone run of this file). This merge reproduces it unchanged. Worth
   an owner eventually; not this merge's problem, and the paired comparison
   is what makes that claim checkable rather than an assertion.

5. **Deviations from the brief, all of them small and each deliberate:**
   - The brief's step 2.4 ("update the wiring test's pinned arity if it pins
     38") was a no-op — **no arity pin exists**; §2 documents the search.
   - Both bypass test helpers got a two-line seed of dev's new flat field
     in addition to the accessor kwarg. Without it, an `object.__new__`
     bypass screen would `AttributeError` the moment a moved body reads the
     accessor — recipe §3's seventh bypass shape, applied to a field dev
     added after the shape was catalogued.
   - The recipe's §20 wave-5 pin-trajectory table was **not** given a row for
     this merge. Checked the precedent first: wave-4's own 106-commit dev
     merge is recorded only in the ratchet file's comment, and §19's table
     stops at its wave-close commit (the post-merge value then appears as
     the *next* wave's "start" row). Following that, not inventing a new
     convention.
   - The report you are reading is committed as a third, documentation-only
     commit, because `.superpowers/` is `.gitignore`d and every other report
     in this series is force-added and tracked. Leaving the wave's final
     evidence as an ignored untracked file would have been the larger
     deviation.

6. **Not verified live.** This is a merge and a doc pass; no UI behaviour
   was exercised by hand. The port's behaviour is covered by dev's own
   `test_clear_finished_prunes_the_stale_outcome_for_a_reused_media_id`,
   which drives a real Pilot through the screen delegator into the ported
   controller block.
