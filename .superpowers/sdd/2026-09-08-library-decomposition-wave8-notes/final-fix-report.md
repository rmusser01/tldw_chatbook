# Wave-8 final fix report — the three review conditions, and the reconciliation merge

**STATUS: ALL THREE CONDITIONS MET; RECONCILIATION MERGE COMPLETE; MERGE-READY.**
All three review conditions are executed and evidenced. The merge is a true
merge commit with all five blame-ignore hashes intact. Both dual-receiver
guards are green with both mutations re-verified to red exactly one leg
each. Every red in the battery is named and proven pre-existing at an
isolated `origin/dev` parent. `preflight.sh` is green — including the
duplicate-id check that the renumber existed to keep green.

**Branch:** `refactor/library-decomp-wave8-notes`
**Status at start:** MERGE-READY WITH CONDITIONS (final whole-branch review)
**Worktree:** `/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation`
**Python:** branch `.venv` 3.14.2; isolated `origin/dev` parent worktree with its own `uv venv`, 3.14.2

| # | Commit | What |
|---|---|---|
| 1 | `19e5aa858` | `fix(backlog): renumber TASK-32041 -> 32047, and two evidence errata` |
| 2 | `07cf9d3f1` | `Merge remote-tracking branch 'origin/dev'` — a real merge commit, two parents (`19e5aa858`, `2c6a7de49`) |
| 3 | `81845a1ba` | `fix(library): reconcile the dual-receiver guard and the screen ratchet with dev` |

`progress.md` was never touched (it stays modified-in-worktree, unstaged, exactly as found).
Nothing pushed.

---

## 1. TASK-32041 renumber — CRITICAL, and the review's suggested id was already wrong

**The collision had fired.** dev's `553960448` ("file critique #7 fix tasks
32041-32046") minted its own `task-32041` — *Library media: at 235 wide,
arrow-Down leaks focus into the reader and Escape will not close it* — and
`973b4c039` closed it. Different filenames, so `git merge` takes both
silently; the duplicate only surfaces as `preflight.sh`'s / CI's
duplicate-id check going red **after** the merge.

**The re-sweep found the review's own suggestion stale.** The review said
"32040 was free at review time — re-verify". It is not free: **this branch
already holds 32040** for the notes-sync cutover-guard filing, minted in the
same commit `bad16ab8f`. Had that been taken on trust, the fix would have
created a second collision inside our own tree.

**The sweep actually run**, immediately before choosing, after `git fetch origin`:

```
git rev-list --objects --all | grep -oE 'task-[0-9]+' | grep -oE '[0-9]+' | sort -rn | head -1
```

One command over every blob path reachable from **all 655** local + remote
refs — so renamed, deleted and merge-commit-only task files are counted,
which is the blind spot that made a `git log --all --diff-filter=A` sweep
lie on 2026-09-04. **True max: 32046. Next free: 32047**, verified free as a
filename *and* as a content reference (`git grep task-32047` across every
ref: no hits).

**The direction of the renumber is the rule's, not convenience.** TASK-19601's
older-keeps-id rule: dev's holder has `created_date` `2026-09-08 14:36`,
ours `2026-09-08 15:04` — **dev's is older by 28 minutes**, so ours moves.
The task file carries a `## Renumbering provenance` section stating this,
per the same rule.

Renamed with `git mv` (rename detected, `R063`), `id: TASK-32041` →
`id: TASK-32047`, and the three inbound references updated:

- `backlog/tasks/task-31249 - …md:59` (the wave-8 census block)
- `backlog/docs/library-decomposition-recipe.md` §25
- `.superpowers/sdd/…/task-4-report.md` §4 — a **visible erratum block**, not a
  history rewrite: the historical text still says 32041 because `bad16ab8f`
  really did file it that way, and the erratum states the mapping, the
  cause, the re-sweep and the lesson.

### Reported, not fixed: TASK-32040 is a latent triple collision

Our 32040 is unique in **both trees being merged here**, so it is not this
merge's problem. But the same all-refs sweep shows **two other, unrelated**
`task-32040` files on unmerged branches:

| id | title | ref |
|---|---|---|
| 32040 | Notes-sync-cutover-guard-is-permanently-vacuous… | **ours** (this branch) |
| 32040 | Console-Assistant-turn-harness-omits-the-split-production-stylesheet | `origin/codex/dev-test-review-20260904` |
| 32040 | Preserve-selected-private-history-when-Console-trace-capture-is-enabled | `origin/codex/kokoro-speech-trace-recovery` |

Those two collide with each other regardless of this branch. Whichever lands
on `dev` first turns ours red. Flagged for whoever merges them; deliberately
not renumbered here, since renumbering ours fixes nothing about the pair.

---

## 2. Probe label — "two filesystems" → "two checkout locations"

**The spot-check is confirmed, live:**

```
df  ~/Documents/…/library-decomp-foundation  ->  /dev/disk3s5  on /System/Volumes/Data
df  /private/tmp                             ->  /dev/disk3s5  on /System/Volumes/Data
stat -f %d  both paths                       ->  16777234 (identical)
```

Same APFS volume, therefore the same mount options (`apfs, local, journaled,
nobrowse, protect, root data`) — no mount-flag difference is even reachable.
The label is false.

**What survives, untouched:** the phenomenon (a 30 ms median swing, 500 ms
[482–702] vs 470 ms [428–490], from **byte-identical** source with `diff -r`
empty), the 118 runs across five trees, the elimination ladder, and the §9
rule that both sides must be probed from scratch worktrees. The rule never
rested on the word "filesystem"; it rests on "same kind of location", which
is what it now says.

**What does not survive is the stated mechanism — and that is the point.**
§24 item 3, *a citation used as a REASON is load-bearing, and a wrong reason
is worse than a thin one*, is this close's own contribution, and it applies
to this close. "Different filesystems" is a mechanism a reader can act on
and would have sent the next investigator somewhere empty. The artifacts now
say the mechanism is **unidentified**, and name candidates as candidates:
per-path Spotlight importer scope; directory metadata and tree density under
a large home-directory checkout; endpoint/AV scanners scoped to user-data
paths.

**One candidate is recorded as ELIMINATED** on the machine that took the
readings, so nobody re-spends a round on it: `~/Documents` there is a plain
local directory (`drwx------`, not a symlink; no
`Library/Mobile Documents/com~apple~CloudDocs/Documents`), so
file-provider/iCloud "Desktop & Documents" sync is off and cannot be the
cause.

Relabelled at every site: recipe §9 (rule + erratum), §23 (the narrative and
the "valid pair" heading), §25 (the lessons list), and the task-4 report's
§7.1, §7.2 and a full erratum block in §8.

---

## 3. Gap count — "six of the seven" is five

Re-derived with the ratchet's own `_measure` semantics (`ast` walk of the
`LibraryScreen` class body, not `wc -l`) over `git show <commit>:library_screen.py`
for **all sixteen** boundary commits. Every cell reproduced to the digit.

| gap | lines | methods | class |
|---|---|---|---|
| w1→w2 | 43965 → 43965 = **+0** | 1282 → 1282 = +0 | **ZERO** |
| w2→w3 | 42411 → 43977 = +1566 | 1267 → 1316 = +49 | positive |
| w3→w4 | 42940 → 43225 = +285 | 1304 → 1311 = +7 | positive |
| w4→w5 | 41155 → 41574 = +419 | 1295 → 1302 = +7 | positive |
| w5→w6 | 40094 → 41393 = +1299 | 1296 → 1321 = +25 | positive |
| w6→w7 | 37574 → 37537 = **−37** | 1282 → 1282 = +0 | **negative** |
| w7→w8 | 34669 → 35777 = +1108 | 1260 → 1290 = +30 | positive |

**Five positive, one zero, one negative.** The zero was being counted as
positive. Everything else survives exactly: the sum is still **+4,640**, the
arithmetic still closes (`45,134 − 17,544 + 4,640 = 32,230`;
`1,300 − 160 + 118 = 1,258`), and the give-back is still **26.4%**.

The recipe's §25 blockquote had the same defect in a second form — it listed
only six values and said "every gap is positive except one", silently
absorbing the zero. Both are corrected, and §25 gains the zero-width-window
footnote, including the detail that the **method** side has *two* zeros
(`0, +49, +7, +7, +25, 0, +30`): w6→w7 is flat in methods while −37 in
lines, i.e. that window was pure body churn inside existing methods.

---

## 4. The reconciliation merge

`git merge origin/dev` — a merge commit, never a rebase or squash. **All five
blame-ignore hashes verified still resolvable and reachable from HEAD** after
the merge:

| hash | commit |
|---|---|
| `ec7318181` | notes state object + shims (notes series 1/3) |
| `9e13f0207` | notes controller + delegators (notes series 2/3) |
| `a529dbd3e` | revert one notes mover; retarget two path censuses |
| `3d670ee38` | notes cleanup (notes series 3/3) |
| `9d8244c72` | seed three media-trash fakes the notes retarget reached |

**99 dev commits**, not the ~87 seen at review time — `origin/dev` advanced
`6ff3dfae2 → 2c6a7de49` during the review window, which is also how the
32041 collision got in. 120 files changed, +29,610 / −586.

**Zero conflicts.** The four files touched on both sides
(`test_library_shell.py`, `library_screen.py`, `canvas_sync.py`,
`production-diagnostic-inventory.json`) all resolved textually.

### 4.1 The guard fix — proven, then re-proven by mutation

Post-merge, **both** legs of `test_media_row_toggle_resolves_the_dotted_state_path`
went red, exactly as the reviewer predicted. Root cause **captured, not
inferred**:

```
AttributeError: '_Screen' object has no attribute 'query'
  at canvas_sync.py:325 ->  bulk_reason = screen.query("#library-media-select-bulk-reason")
```

dev's task-32045 added that call to the media leg; the dispatcher's own
`except Exception` swallows the `AttributeError` and fires the
`screen.refresh(recompose=True)` the guard asserts must never fire.

**Fix:** a minimal `query` stub returning an **empty** result on both legs'
doubles. Empty is the honest shape — dev's own `if bulk_reason:` already
handles the reason line being absent — and it keeps the test asserting
exactly what it guards: that the dotted `_media_state` path resolves on this
receiver. **The `refresh`-raises assertion was not weakened**; it is the
entire mutation surface of the wave's one behaviour fix.

**The notes guard needed nothing**, re-verified rather than assumed: both its
legs already stub `query` (returning `()` for anything that is not
`.library-notes-row`), so any new lookup in the notes leg is absorbed.

**BOTH MUTATIONS RE-RUN AFTER THE RECONCILIATION.** Each still reds exactly
one leg, in opposite directions:

| mutation | red | green | verdict |
|---|---|---|---|
| *(none — post-merge, post-stub)* | — | **4 / 4** | baseline |
| remove `LibraryMediaController._media_state` accessor | **`[controller]`** only | 3 | ✅ exactly one leg |
| revert the media dotted branch → flat `f"_library_{kind}_row_selection"` | **`[screen]`** only | 3 | ✅ exactly one leg |

Production files restored byte-clean after each mutation (`git diff` empty).
The deliberate change survived the reconciliation intact.

### 4.2 `test_library_shell.py` — the predicted hotspot, and the union is exact

dev +78/−2 (2 new tests) against our 342/342 retargets. Resolved textually,
and the union is provably exact **in both directions**:

| comparison | result | meaning |
|---|---|---|
| post-merge vs **ours** (`19e5aa858`) | +78 / −2 | exactly dev's contribution, nothing more |
| post-merge vs **dev** (`origin/dev`) | +342 / −342 | exactly our retargets, nothing more |

Def counts: merge-base 871, `origin/dev` 873, post-merge **873**. Dev's two
new tests (`test_media_list_arrow_keys_move_the_cursor_at_every_width`,
`test_slash_focuses_the_media_filter_not_the_rail_search`) are both present;
**0** dev-added defs lost, **0** of ours lost. Nothing was duplicated.

### 4.3 `library_screen.py` — dev's +33, and no mover body was touched

Dev's two hunks, both verified rather than assumed:

| task | lines | enclosing method | moved by us? |
|---|---|---|---|
| 32046 | +23 | `on_key` | **no** — screen-resident |
| 32039 | +10 | `on_screen_resume` | **no** — screen-resident |

Neither enclosing method is among the 26 names this wave removed from
`LibraryScreen`, so **nothing needed porting into a controller**. Both
methods are **byte-identical to `origin/dev`'s version** post-merge
(AST-extracted and compared), i.e. dev's edits landed whole and we had not
touched either method. Dev added **zero** new `LibraryScreen` methods since
the merge-base, so no dev-added method could be lost — verified 0 lost.

### 4.4 Censuses over dev's drift

**Six-spelling census, 100 notes field names** (derived authoritatively from
`dataclasses.fields(LibraryNotesState)` through `notes_state_shim_attr()` —
100 fields, 100 distinct flat names).

The instrument was **proven on a positive control before being trusted**: it
finds `canvas_sync.py:238`'s computed-name f-string (the fifth spelling), the
one hit the recipe accounts for. Two earlier drafts were discarded for being
wrong in opposite directions — a loose substring match produced false
positives (`f"{destination}_reader"`), and a naive producible-name regex made
the row degenerate (`f"{shortened}{suffix}"` matches every name that exists).

| file | attribute | `LibraryScreen.<n>` | quoted | kwarg | `def` | `ast.Name` | f-string | total |
|---|---|---|---|---|---|---|---|---|
| `library_screen.py` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| `test_library_shell.py` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

**"`library_screen.py` carries ZERO flat notes references in any spelling"
holds through the merge**, and dev's additions spell zero flat notes names.

**Whole-tree residue across all six roots: 467** — *exactly* the accounted
set, unchanged by the merge (410 controller shim loop + 17 frozen evidence
script + 13 + 3 cross-subsystem accessor properties + 10 known false positive
+ 9 + 3 wiring-test pins + 1 `canvas_sync.py` + 1 cutover guard). A 468th
apparent hit was run down and is a **false positive of my deliberately
broadened f-string heuristic on a PRE-EXISTING line** — `Prompts_DB.py:3610`,
`row[f"{column}_preview"]`, a sqlite `Row` key rather than an attribute, on a
commit that is an ancestor of the merge-base and therefore not dev drift at
all.

**Pruned-name check, bound AND unbound** — the 26 names this branch removed
from `LibraryScreen`, censused across all six roots in every spelling:
**116 hits, all in exactly two expected places** — their destination
(`library_notes_controller.py`: the `def`s plus their own `self.<name>`
calls) and `test_library_notes_wiring.py`'s literal pins, which are the guard
that asserts the move. **Zero unbound `LibraryScreen.<name>`. Zero from
`library_screen.py`. Zero from any dev-touched file.**

### 4.5 Ratchet rows and the diagnostic inventory

`library_screen.py` re-pinned **32230 → 32263 lines**, methods unchanged at
**1258** — the +33 is entirely dev's, in screen-resident methods (§4.3),
re-pinned per `test_budget_is_not_left_slack_after_a_wave`'s own instruction
("set it to {lines} so the gain is locked in"). Measured post-merge: 32263
lines / 1258 methods / **1251 distinct names** — the 7-name gap is §25's
duplicate-collapse finding, independently reproduced here.

**A finding worth dev's attention: `origin/dev` is RED on its own
`library_screen.py` ratchet today.** 35810 lines against its own pin of
35777, `+33` — dev landed those 33 lines without re-pinning. Pre-existing,
not created by this merge, and **this merge fixes it**: the branch carries 4
ratchet reds post-merge where `origin/dev` carries 5.

**Diagnostic inventory: GREEN post-merge with no drift**, so per protocol
(read the rows first, only then regenerate) there was nothing to read and
nothing to regenerate: `592 owners, 1348 TASK-492 calls, 55 TASK-31551 calls,
7681 TASK-494 calls, 12 sink files`.

---

## 5. Battery

All runs `-p no:randomly`, branch `.venv` Python 3.14.2. Dismissals proven on
an isolated `origin/dev` worktree (`2c6a7de49`) with its **own** `uv venv`.

**Parity proven, and a trap avoided.** Interpreter parity 3.14.2 both. The
parent venv's editable finder maps to its own tree — but a first check run
from the *branch's* cwd resolved the **branch** tree, because Python puts the
cwd at the front of `sys.path` and it wins over the editable finder. Every
parent run below therefore `cd`s into the parent worktree first; verified
`tldw_chatbook.__file__` resolves under the scratch path from there.

| chunk | branch | parent (`origin/dev`) | verdict |
|---|---|---|---|
| A — 10 wiring suites + both size ratchets + **both dual-receiver guards** + ui-ready census | **5 failed / 139 passed**, 16.07s | ratchets subset: **5 failed / 41 passed** (branch ratchets alone after the re-pin: 4 failed / 44 passed) | reds all pre-existing; branch is one red BETTER |
| B — recompose ratchet + preimport + screen-reuse + modal-dismissal + adaptive-reader | **4 failed / 543 passed**, 230.39s | **4 failed / 543 passed**, 231.35s | **identical, name for name** |
| C — 7 characterization suites + dev's media render fixes | **11 failed / 143 passed**, 174.88s | **11 failed / 140 passed**, 170.49s | identical 11; +3 = our notes characterization file |
| D — `test_library_shell.py` (the hotspot), 842 tests in 6 chunks | **229 failed / 613 passed** | **227 failed / 615 passed** | 5 of 6 chunks **identical name-for-name**; the 2 difference is inside a measured flake band |
| `./scripts/preflight.sh` | **all checks passed** | | |

### Chunk D in detail — a large PRE-EXISTING red block, and zero branch regressions

`test_library_shell.py` was **not** in the wave-8 close's own battery; it enters
this one only because it is a file dev changed. It turns out to carry **227
failures out of 842 on `origin/dev` itself**.

A single 842-test process could not be trusted here — the first attempt ran
60+ minutes, was killed by harness cleanup at ~89%, and produced no summary
line. Split into 6 chunks of ~141 and run on both trees:

| chunk | branch | parent | branch-only | parent-only | verdict |
|---|---|---|---|---|---|
| aa | 19 / 122 | 19 / 122 | 0 | 0 | **identical set** |
| ab | 27 / 114 | 27 / 114 | 0 | 0 | **identical set** |
| ac | 7 / 134 | 5 / 136 | 2 | 0 | flake band — see below |
| ad | 76 / 65 | 76 / 65 | 0 | 0 | **identical set** |
| ae | 21 / 120 | 21 / 120 | 0 | 0 | **identical set** |
| af | 79 / 58 | 79 / 58 | 0 | 0 | **identical set** |
| **total** | **229 / 613** | **227 / 615** | **2** | **0** | |

**Five of six chunks match name-for-name, not merely by count** — 222 of the
229 reds are provably the same tests on both trees.

**The chunk-`ac` difference is flake, established by measurement rather than
asserted.** The two branch-only names are
`test_library_media_page_error_retains_rows_and_gates_unsafe_controls` and
`test_library_media_stale_page_disables_actions_across_recompose`. Both
**pass in isolation on both trees**. Their assertions are mount/paint settle
failures — `textual.css.query.NoMatches: No nodes match
'#library-media-next'` and `#library-notes-row-0 never mounted within 30.0s
(1025 polls)` — the same family as TASK-31249's documented
`test_library_file_notes_workspace.py` race, not value mismatches. Rerunning
chunk `ac` gave 6 rather than 7 on the branch, so the count is not stable.

**Rates, 5 interleaved branch/parent pairs of chunk `ac`** (interleaving so
machine drift and any background load hit both arms equally — recipe §9's
own discipline):

| test | branch | parent |
|---|---|---|
| `…media_page_error_retains_rows_and_gates_unsafe_controls` | **4/5** | **3/5** |
| `…media_stale_page_disables_actions_across_recompose` | **0/5** | **0/5** |
| chunk total, per run | 7, 5, 6, 6, 6 | 5, 6, 5, 5, 5 |

Overlapping bands, and the parent came out **worse than the branch** on one
pair (parent 6 vs branch 5). The second test never recurred at all in ten
further runs across both trees. Per the recipe's disposition rule this is
recorded as **rates, not a verdict**: a paint settle race present on both
trees, belonging on TASK-31249's census — **not** a regression from this
branch. Dispositioned per §7's ladder rather than dismissed by argument.

### The reds, §7 BY NAME

Every red below is **named** and **proven at the isolated `origin/dev`
parent**, not argued.

| # | test | proven on parent | owner |
|---|---|---|---|
| 1 | `test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[chat_screen.py]` | ✅ | dev, documented |
| 2 | `test_screen_size_ratchet.py::test_task_22507_4_does_not_worsen_chat_screen_base` | ✅ | dev, documented |
| 3 | `test_library_modules_size_ratchet.py::test_controller_does_not_grow_past_its_budget[library_media_browse_controller.py]` (371-pin) | ✅ | dev, documented |
| 4 | `test_library_modules_size_ratchet.py::test_budget_is_not_left_slack_after_a_move[library_conversations_controller.py]` (slack mirror) | ✅ | dev, documented |
| 5 | `test_library_selection_updates.py::test_tier1_toggle_falls_back_to_recompose_on_query_one_failure` | ✅ | dev, documented |
| 6 | `test_library_recompose_ratchet.py::test_library_screen_whole_screen_recompose_count_is_ratcheted` (66/63) | ✅ | TASK-31249, documented |
| 7 | `test_library_screen_reuse.py::test_on_screen_suspend_stops_every_timer_in_isolation` | ✅ | TASK-31249, documented |
| 8 | `test_library_modal_dismissal.py::test_library_modal_inventory_matches_declared_edges_bidirectionally` | ✅ | TASK-31815, documented |
| 9 | `test_screen_preimport.py::test_screen_preimport_scheduled_only_after_ui_ready_flips` | ✅ | **dev, NOT previously documented — new in the 99 commits** |
| 10–20 | `test_library_media_render_fixes.py` — 11 tests | ✅ all 11, identical names | **dev's own file, all 11 red on `origin/dev`** |
| 21–247 | `test_library_shell.py` — **227** tests | ✅ 222 by identical NAME across 5 of 6 chunks; the remaining band rate-measured (4/5 vs 3/5, and 0/5 vs 0/5) | **dev, pre-existing — 227/842 red on `origin/dev` itself** |

`test_screen_size_ratchet.py::…[library_screen.py]` is **red on the parent
and GREEN on the branch** — the merge fixes a dev-side red (§4.5).

Both dual-receiver guards (4 legs) are **GREEN on the branch**, with the
mutation table in §4.1. On `origin/dev` the single-leg media guard is **red**
— see §6.

---

## 6. To flag to dev

1. **dev's own media toggle guard is red on `origin/dev` today.** Confirmed
   by running it there: `Tests/UI/test_library_selection_updates.py::test_media_row_toggle_resolves_the_dotted_state_path`
   fails (2 failed / 5 passed in that file). Same mechanism as ours — dev's
   task-32045 added `screen.query("#library-media-select-bulk-reason")` to
   the media leg but never stubbed `query` on that test's own double, so the
   `AttributeError` is swallowed and the fallback `refresh` fires. dev's
   version is single-leg (no `receiver_kind` parametrization). **This merge
   fixes it** and additionally gives it the `[controller]` leg.
2. **`origin/dev`'s `library_screen.py` size ratchet is red on dev** (35810
   vs its own pin 35777) — the 33 lines from tasks 32046/32039 landed
   without a re-pin. Fixed by this merge.
3. **`test_screen_preimport_scheduled_only_after_ui_ready_flips`** is a new
   dev-side red that arrived in the 99 commits; nobody owns it yet.
4. **`Tests/UI/test_library_shell.py` is 227-red out of 842 on `origin/dev`**
   — 27% of the file. It was never in this program's battery (it is not a
   decomposition guard), so the program never surfaced it; it is visible here
   only because dev changed the file. Unowned, unfiled, and by far the largest
   red block found in this reconciliation.
5. **The latent TASK-32040 triple collision** on two unmerged `codex/`
   branches (§1).

---

## 7. Concerns

- **The 32040 latent collision (§1)** is the only unresolved id risk, and it
  is not resolvable from this branch.
- **Two large unowned dev-side red blocks**, neither this branch's and
  neither previously filed: `test_library_media_render_fixes.py` at 11 red
  (dev's own newest media work failing its own tests), and
  `test_library_shell.py` at **227 of 842**. The second is the more serious:
  it is the Library shell's own test file, it is 27% red on `dev`, and the
  only reason this reconciliation saw it is that dev happened to touch the
  file. A guard nobody runs is a guard that is not there.
- **One 842-test file cannot be run as a single process here** and be
  trusted. The first attempt took 60+ minutes, was killed at ~89% by harness
  cleanup, and emitted no summary line — a run that would have been easy to
  misreport from its partial `F` stream. Chunking into ~141-test slices made
  every chunk finish, produced comparable numbers on both trees, and is what
  turned "hundreds of Fs" into "222 identical names plus a measured flake
  band". Recorded because the next reconciliation will meet the same file.
- **The id-sweep lesson generalises and is now filed** in
  `lessons-backlog-hygiene.md`: a correct sweep is a snapshot, not a
  reservation; the longer the branch, the more certainly it is stale. The
  durable backstop is running `preflight.sh` **after** the reconciliation
  merge, since the duplicate-id check can only see a collision once both
  sides are in one tree — a green preflight before merging says nothing
  about this failure mode.
- **The probe mechanism is still unknown (§2).** The rule is safe and the
  measurement is sound, but the *cause* of a reproducible 30 ms
  location-dependent swing on byte-identical source is unexplained. It is
  now labelled as unexplained rather than mislabelled, which is the most
  this close can honestly claim.

## Round-2 erratum (coordinator): the collision struck TWICE
Dev's next 30 commits minted BOTH task-32040 and task-32047, colliding with both of this wave's filings. Renumbered (ours, unmerged side): 32040 -> **TASK-32088** (cutover guard), 32047 -> **TASK-32089** (dispatcher guard). Two rule refinements recorded in lessons-backlog-hygiene: landed-keeps-id trumps older-keeps-id; renumber as the last commit before push, never mid-fix-wave. Dev's identically-numbered tasks (Console trace capture; PII log redacting) are untouched, as are lessons-testing-evidence:12295 and decisions/029 which cite DEV's tasks.
