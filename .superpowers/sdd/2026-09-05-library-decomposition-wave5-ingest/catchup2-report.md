# Round-2 dev-reconciliation report — `origin/dev` merge (72 commits, TASK-31521)

Executes `catchup2-brief.md` against worktree
`/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation`,
branch `refactor/library-decomp-wave5-ingest`, starting at HEAD `075c5c35a`
with a `git merge origin/dev` already in progress.

Nothing was pushed. No rebase, no squash, no abort. `progress.md` was never
staged or edited (it carried a pre-existing unstaged modification when this
session started and still does).

---

## 0. Merge state, independently verified

Every figure below is from `git` output in this session, not from the brief.

```
$ git rev-parse --abbrev-ref HEAD        -> refactor/library-decomp-wave5-ingest
$ git rev-parse HEAD                     -> 075c5c35afe2f8c99b2932c01547201ed63e8f78
$ git rev-parse MERGE_HEAD               -> 2c9c144181b942af2d29d16b9eb2681d7f5a7212
$ git rev-parse origin/dev               -> 2c9c144181b942af2d29d16b9eb2681d7f5a7212
$ git merge-base HEAD MERGE_HEAD         -> 93388ba69b7499c2bc3180fc26c82d7f341871a7
$ git rev-list --count 93388ba69..MERGE_HEAD -> 72
```

Merge-base `93388ba69` is the previous reconciliation's `origin/dev` tip, so
the two rounds chain exactly. `MERGE_HEAD == origin/dev`, so `git show
origin/dev:<path>` is a valid source for dev's side.

Conflicts:

```
$ git diff --name-only --diff-filter=U
tldw_chatbook/UI/Screens/library_screen.py

$ grep -n '^<<<<<<<\|^=======$\|^>>>>>>>' tldw_chatbook/UI/Screens/library_screen.py
3460:<<<<<<< HEAD      3461:=======      3527:>>>>>>> origin/dev
20533:<<<<<<< HEAD     20535:=======     20748:>>>>>>> origin/dev
20755:<<<<<<< HEAD     20757:=======     20796:>>>>>>> origin/dev
```

**One file, three hunks — exactly as the brief states.** No other conflict of
any kind, so no "resolve mechanically toward dev" case arose in a non-library
file. Total merge size: 185 files changed, 30182 insertions(+),
28141 deletions(-).

---

## 1. Verifying the brief's analysis before applying it

### 1a. Dev edited exactly TWO moved bodies

Reproduced the brief's AST claim independently, by extracting **every**
controller-resident method name and comparing its `LibraryScreen` body text
between merge-base and `origin/dev`:

```
controller methods: 110
present on both screens, IDENTICAL : 80
present on both screens, CHANGED   : 4 -> ['__init__', 'refresh',
                                           '_handle_library_ingest_registry_changed',
                                           '_handle_library_ingest_progress_changed']
not on merge-base screen (controller-only/new): 26
on base screen but GONE from dev screen: 0
```

`__init__` and `refresh` are **name collisions, not moved bodies** — they are
the controller's own constructor and its screen-forwarding proxy:

```python
    def refresh(self) -> Any:
        return self._screen.refresh
```

So the real answer is **exactly the two bodies the brief names**. Confirmed.

### 1b. The two edits are exactly what the brief describes

`diff -u` of each body, merge-base vs `origin/dev`:

**`_handle_library_ingest_registry_changed`** — two edits, matching the brief
item-for-item:

- (a) the `LIBRARY_ROW_INGEST_MEDIA` block gains a suspended gate
  (`if self._library_screen_suspended: … = True` / `else:` the original
  dynamic-regions + shortcuts block, re-indented, with a 3-line comment);
- (b) `if grew:` becomes `if grew and not self._library_screen_suspended:`
  with its 3-line comment, and the landing-attention gate becomes a
  parenthesised `not self._library_selected_row_id and not
  self._library_screen_suspended`.

**`_handle_library_ingest_progress_changed`** — the entry guard is
restructured exactly as described: `is_attached` first, then a suspended-gate
block that latches `_library_ingest_suspended_activity` when the ingest row
is selected and returns, then the row check.

### 1c. The 4 new `__init__` fields really are additions

The brief's key claim — that the `_library_ingest_path_debounce_timer` line
and every other moved-field line on dev's side of hunk 1 are *context*, not
additions. Verified against `git diff -U0 93388ba69 origin/dev`: in the init
region the only `+` lines are the 16 belonging to the 4 new fields. The
`_library_ingest_path_debounce_timer` name appears as a `+` line **only** at
dev's new `on_screen_suspend` tuple, never in `__init__`.

---

## 2. Hunk resolutions

The replacement text for hunk 1 was **derived programmatically from
`git diff -U0 93388ba69 origin/dev` output**, not hand-transcribed: the
resolver script selects the diff hunk whose additions contain
`_library_visit_entered`, strips the `+` markers and writes those lines
verbatim. Transcript:

```
hunk-1 replacement: 16 lines from git diff
   |        #: The 5s first-load failsafe armed in ``on_mount``; retained so
   |        #: ``on_screen_suspend`` can stop it (Qodo #2414 finding 3).
   |        self._library_source_snapshot_timeout_timer: Timer | None = None
   |        #: TASK-31521 (screen reuse): True while this screen is covered by
   |        #: another. Gates the ingest listener's DOM/DB branches -- the
   |        #: listener itself stays registered across suspend (its counting and
   |        #: cross-tab toast are ambient signals), but widget rebuilds and
   |        #: snapshot re-reads are deferred to one resume-time pass.
   |        self._library_screen_suspended: bool = False
   |        #: Set when a gated registry event fired while suspended, so resume
   |        #: runs exactly one ingest-UI reconciliation instead of N.
   |        self._library_ingest_suspended_activity: bool = False
   |        #: False until the first ScreenResume finishes its surface kicks;
   |        #: distinguishes entry semantics (pristine trash state, entry focus)
   |        #: from repeat-visit refresh (preserve the user's live context).
   |        self._library_visit_entered: bool = False

hunks (0-based start/mid/end): [(3459, 3460, 3526), (20532, 20534, 20747),
                                (20754, 20756, 20795)]
hunk at 20532: keeping HEAD ->  return self._ingest_controller._handle_library_ingest_registry_changed()
hunk at 20754: keeping HEAD ->  return self._ingest_controller._handle_library_ingest_progress_changed(before, after)
written. remaining markers: 0
```

The script asserted, before writing, that hunk 1's HEAD side is empty and
that hunks 2 and 3's HEAD sides are each exactly one line beginning
`return self._ingest_controller.` — so a mis-located hunk would have failed
loudly rather than silently resolving wrong.

| Hunk | Side taken | Result |
|---|---|---|
| 1 (init, ~3460) | dev's **4 new fields only** | 12 moved `LibraryIngestState` field lines dropped; the 16 kept lines are in dev's own order |
| 2 (~20533) | **HEAD** delegator | dev's edit ported to the controller (§3) |
| 3 (~20755) | **HEAD** delegator | dev's edit ported to the controller (§3) |

`ast.parse` on the resolved file: **parses OK**.

---

## 3. Porting the two edits into the controller

Both edits were applied to `LibraryIngestController`'s copies of the two
bodies. **Byte-for-byte proof** — AST body-range extraction, `origin/dev`'s
`LibraryScreen` methods vs the merged tree's `LibraryIngestController`
methods:

```
=== _handle_library_ingest_registry_changed ===
  IDENTICAL (byte-for-byte, zero substitutions needed): True
=== _handle_library_ingest_progress_changed ===
  IDENTICAL (byte-for-byte, zero substitutions needed): True
```

**Zero accessor substitutions were needed at the call sites.** The
established accessor shape is a same-named `@property` (plus `@x.setter`
where a body writes), so `self._library_screen_suspended` and
`self._library_ingest_suspended_activity = True` read and write identically
on the controller and on the screen. That is stronger than the brief's
"identical modulo the accessor substitutions" bar: the ported text needed no
modulo at all.

---

## 4. The three new accessors

### Measured arity — the brief asked for real numbers, and its own were low

```
BEFORE:  total params (incl self+screen): 40    keyword-only: 38
AFTER:   total params (incl self+screen): 43    keyword-only: 41
```

The brief said "38 → 40 keyword-only". The real figure is **38 → 41**: the
brief counted the getter/setter pair as one parameter, but the established
pattern passes the getter and the setter as two separate keyword arguments
(`library_canvas_resync_pending_accessor` + `set_library_canvas_resync_
pending` is the precedent in this very constructor). Three parameters were
added, so **41 keyword-only / 43 total**. Both ratchet comments and the
merge-commit message carry the measured figure.

### The three bindings

| Parameter | Shape | Why |
|---|---|---|
| `library_screen_suspended_accessor` | getter only | ported code only reads it; `on_screen_suspend`/`on_screen_resume` own the writes. Screen-wide lifecycle state (dev gates media and notes on it too), so this accessor is its **permanent** shape — the `library_canvas_resync_pending_accessor` standing |
| `library_ingest_suspended_activity_accessor` | getter | both ported bodies read/write the latch |
| `set_library_ingest_suspended_activity` | setter | required: both ported bodies **set** it True. Ingest-exclusive, so an **interim** bridge like analyze_outcomes — task-31651 retires both |

Properties added mirror the skills controller's precedent exactly
(`@property` returning `self._<n>_accessor()`; `@<n>.setter` calling
`self._set_<n>_fn(value)`).

### All three construction sites wired

`grep -rn 'LibraryIngestController(' tldw_chatbook/ Tests/` finds exactly
three, all updated:

| Site | Bindings added |
|---|---|
| `tldw_chatbook/UI/Screens/library_screen.py:2593` (production) | all 3, `lambda: self._…` / `setattr(self, …)` |
| `Tests/UI/test_library_shell.py` `wire_bypass_ingest_controller` | all 3, `lambda: screen._…` / `setattr(screen, …)` |
| `Tests/UI/test_library_ingest_inline_consent.py` `_wire_bypass_ingest_controller` | all 3, same |

### Bypass seeds

Both bypass helpers seed the two flat fields, matching `__init__`'s defaults
— which were read out of the source rather than assumed:

```
  __init__ default: self._library_screen_suspended: bool = False
  __init__ default: self._library_ingest_suspended_activity: bool = False
```

```python
    if not hasattr(screen, "_library_screen_suspended"):
        screen._library_screen_suspended = False
    if not hasattr(screen, "_library_ingest_suspended_activity"):
        screen._library_ingest_suspended_activity = False
```

Same two-helper pattern as the analyze_outcomes seeds — recipe §3's seventh
bypass shape, applied to fields dev added after the shape was catalogued.

### Module docstring

- Group (b) count **8 → 10**, with both new names and their read vs
  read+write placement.
- The divergence paragraph is rewritten from *"One body follows a post-move
  dev edit"* to **"Three bodies follow post-move dev edits (reconciliation
  merges)"**, enumerating all three, naming **TASK-31521** alongside
  task-28007, distinguishing the one permanent accessor from the two interim
  bridges, and pointing at task-31651 for the exit (now including the
  `on_screen_suspend` seam).

---

## 5. `on_screen_suspend` — dev's timer-stop fix no-ops on this branch

Dev's new `on_screen_suspend` (auto-merged, **not** conflicted) stops timers
through a string loop whose tuple included `"_library_ingest_path_debounce_
timer"`. Verified that on this branch that name is not a screen attribute:

```
$ grep -n '_library_ingest_path_debounce_timer' tldw_chatbook/UI/Screens/library_screen.py
3465:  (dev's side of conflict hunk 1 — removed by the resolution)
9577:  (the on_screen_suspend string loop)

$ grep -n 'path_debounce_timer' tldw_chatbook/UI/Library_Modules/library_ingest_state.py
148:    path_debounce_timer: Timer | None = None
```

The flat shim was deleted at ingest cleanup, so `getattr(self, attr, None)`
silently returns `None` and **dev's fix no-ops for the ingest timer**. This
is dev's own new screen-resident method (not a moved body), so it was edited
here: the name leaves the tuple and the timer gets an explicit state-object
stop with the same stop+`None` shape and a comment recording why.

```python
        # The ingest path-debounce timer lives on ``_ingest_state``, not as a
        # flat screen attribute, so the string loop above cannot reach it --
        # a ``getattr`` for the old flat name silently returns None.
        ingest_timer = self._ingest_state.path_debounce_timer
        if ingest_timer is not None:
            ingest_timer.stop()
            self._ingest_state.path_debounce_timer = None
```

---

## 6. Census over the MERGED tree

21 names (the 20 `LibraryIngestState` fields + `_library_ingest_analyze_
outcomes`), in **both** the attribute form and the quoted-string form, over
`tldw_chatbook/**` + `Tests/**`, excluding the controller and state modules.

```
21 names censused (20 state fields + analyze_outcomes)
TOTAL: 17
```

| Hit | Form | Disposition |
|---|---|---|
| `tldw_chatbook/Library/library_ingest_state.py:1215` | ATTR | **Not a use** — prose in a docstring in the unrelated pure-logic module. Same disposition as last round |
| `library_screen.py` ×4 + `test_library_ingest_analyze_skipped.py` ×3 + both bypass helpers ×5 | ATTR/STR | All `_library_ingest_analyze_outcomes`, which is **not** a state field — it is the flat screen field bridged by its accessor. Correct as-is |
| **`Tests/UI/test_library_screen_reuse.py:111` and `:212`** | **STR** | **REAL — retargeted, see below** |

### The string-form hits the attribute grep would have missed

Dev's new TASK-31521 test file enumerates
`"_library_ingest_path_debounce_timer"` as a **string** in two timer tables.
This is precisely the blindness the brief predicted, and it was not
theoretical:

```
$ .venv/bin/python -m pytest Tests/UI/test_library_screen_reuse.py -p no:randomly -q
FAILED …::test_on_screen_suspend_stops_every_timer_in_isolation
1 failed, 3 passed
E   AttributeError: 'LibraryScreen' object has no attribute '_ingest_state'
```

Two mechanical retargets, same receiver-swap family as last round's
`clear_finished_armed_at` fix, each with a one-line reason comment:

1. **`test_library_reuse_and_suspend_timer_quiescence` (line ~111)** — the
   flat name in the `getattr(library, attr, None) is None` table was a
   **vacuous pass** (a missing attribute trivially reads `None`). Removed
   from the tuple and replaced with an explicit
   `assert library._ingest_state.path_debounce_timer is None`.
2. **`test_on_screen_suspend_stops_every_timer_in_isolation` (line ~212)** —
   an `object.__new__` bypass screen. The flat name is removed from
   `timer_attrs`; the test now seeds `screen._ingest_state =
   LibraryIngestState()` (the established fixture pattern), arms
   `path_debounce_timer` with the same `_RecordingTimer`, and asserts it was
   both stopped and cleared. The test's "a new timer added without updating
   this table fails loudly" property is preserved — all seven are still
   enumerated, six flat and one on the state object.

```
$ .venv/bin/python -m pytest Tests/UI/test_library_screen_reuse.py -p no:randomly -q
4 passed
```

### The 4 new fields, swept separately

`grep -rn` for `_library_screen_suspended|_library_ingest_suspended_activity|
_library_visit_entered|_library_source_snapshot_timeout_timer` across
`Tests/` returns **only this session's own additions** (the seeds, the
accessor lambdas, one ratchet comment) plus `test_library_screen_reuse.py`.
No other test touches them, so no further bypass seeding is needed.

---

## 7. Ratchet re-pins (in the merge commit)

Both re-measured with each file's own `_measure()` on the merged tree.

| Row | Before | After | Delta |
|---|---|---|---|
| `test_screen_size_ratchet.py` — `LibraryScreen` | `(41028, 1313)` | **`(41371, 1321)`** | +343 / +8 |
| `test_library_modules_size_ratchet.py` — ingest controller | `2623` | **`2721`** | +98 |

### The method delta is the load-bearing check

```
merge-base 93388ba69 : (42518, 1319)
origin/dev 2c9c14418 : (42864, 1327)     -> dev's own delta: +346 / +8
branch     075c5c35a : (41028, 1313)
merged               : (41371, 1321)     -> 1313 + 8 = 1321 exactly
```

An exact `+8` means the resolution re-added **no** moved body and lost **no**
wave-5 delegator — the two failure modes a naive resolution produces.

### Line composition reconciles exactly

`343 = 346 − 20 + 11 + 6`:

- `346` — dev's own line delta on this class;
- `−20` — the lines dev spent editing the two **moved** bodies, which this
  branch takes on the *controller* instead (and which is exactly the ported
  total below, so the two figures cross-check);
- `+11` — the three new accessor bindings at the production construction site;
- `+6` — the `on_screen_suspend` fix (−1 tuple entry, +7 explicit stop).

### Controller `+98`, itemised from `git diff` hunk headers

| Hunk | Delta | What |
|---|---|---|
| `@@ -342,17 +342,21` | +4 | docstring group (b) 8 → 10 |
| `@@ -381,23 +385,43` | +20 | docstring divergence paragraph, one body → three |
| `@@ -522,11 +546,26` | +15 | 3 constructor params + comments |
| `@@ -614,9 +653,16` | +7 | storing them |
| `@@ -749,6 +795,20` | +14 | `_library_screen_suspended` property |
| `@@ -761,6 +821,24` | +18 | `_library_ingest_suspended_activity` property + setter |
| `@@ -1051,11 +1129,19` | +8 | ported registry suspend gate |
| `@@ -1166,12 +1252,18` | +6 | ported `grew` + landing-attention gates |
| `@@ -1190,10 +1282,16` | +6 | ported progress entry guard |

Sum **= 98** ✓. Ported total **= 20**, matching dev's own +20 — the
arithmetic proof that the port added nothing and dropped nothing.

Both comments follow the files' established dev-merge convention (dated
entry, commit count, merge-base SHA, fresh measure, itemised reason).

### Dev's own pin moves were taken, not overwritten

```
$ diff <(dev's pin rows) <(merged pin rows)
test_screen_size_ratchet.py:            only the library_screen.py row differs
test_library_modules_size_ratchet.py:   only the library_ingest_controller.py row differs (ours; absent on dev)
```

Every row that is not ours is byte-identical to dev's.

---

## 8. TASK-31651 update

Appended via the CLI (repeated `--ac`, so each is individually checkable):

- **#9** — `_library_ingest_suspended_activity` becomes a `LibraryIngestState`
  field and **both** its interim bindings (accessor **and**
  `set_library_ingest_suspended_activity`) are removed, with the note that
  `library_screen_suspended_accessor` explicitly stays (permanent shape).
- **#10** — `on_screen_suspend` no longer needs its out-of-loop
  `_ingest_state.path_debounce_timer` special case, and the suspend-hook
  timer enumeration is expressed one way instead of two.

The Description gained a paragraph explaining the round-2 origin of the
second bridge and scoping the third accessor **out**.

---

## 9. Verification battery

`.venv/bin/python -m pytest`, `-p no:randomly`, from the worktree.

### 9.1 — 6 wiring suites + both ratchets + recompose census guard + support-layer surface

```
3 failed, 81 passed in 48.35s
```

The 3 failures are pre-existing `origin/dev` reds, and this round proves it
**structurally** rather than by re-running dev:

| Row | File vs `origin/dev` | Pin vs `origin/dev` |
|---|---|---|
| `chat_screen.py` budget | **identical** | **identical** (`16966, 563`) |
| `test_task_22507_4_does_not_worsen_chat_screen_base` | **identical** | **identical** |
| `library_media_browse_controller.py` | **identical** | **identical** (`371`) |

Both the measured file and the pin come wholly from dev, so this branch
cannot have caused them and raising the pins would absorb dev's un-owned
creep. Same disposition and reasoning as last round's concern #1. **Our two
re-pinned rows pass.**

### 9.2 — characterization, inline-consent, ingest, screen-reuse

```
Tests/UI/test_library_ingest_characterization.py
Tests/UI/test_library_collections_characterization.py
Tests/UI/test_library_conversations_characterization.py
Tests/UI/test_library_export_characterization.py
Tests/UI/test_library_ingest_inline_consent.py
Tests/UI/test_library_ingest_analyze_skipped.py    (dev's task-28007 file)
Tests/UI/test_library_screen_reuse.py              (dev's TASK-31521 file)
Tests/UI/test_screen_reuse.py                      (dev's TASK-31521 file)
  -> 101 passed in 112.96s
```

**All green.** Notably `test_library_ingest_canvas.py`'s two failures from
last round are **gone** — dev fixed them in these 72 commits.

### 9.3 — dev's other touched library files

```
Tests/UI/test_library_ingest_canvas.py
Tests/UI/test_library_adaptive_reader_shell.py
Tests/UI/test_library_entry_compose_once.py
Tests/UI/test_library_multiselect_media.py
  -> 3 failed, 352 passed in 249.14s
```

All 3 failures are in `test_library_entry_compose_once.py` — see §10.

### 9.4 — direct proof of the port and both accessors

Dev's own `test_suspended_library_gates_ingest_dom_work_until_resume` passes
**unmodified**, and it is end-to-end proof of every piece of this merge's
ingest work. It drives a real `Pilot`, suspends the Library, then:

- calls `library._handle_library_ingest_registry_changed()` — the screen
  delegator into the **ported controller body**;
- asserts `dynamic.call_count == 0` — the ported gate read
  `_library_screen_suspended` through `library_screen_suspended_accessor`
  and took the suspended branch;
- asserts `library._library_ingest_suspended_activity is True` — the ported
  `self._library_ingest_suspended_activity = True` travelled through the
  controller's property setter → `set_library_ingest_suspended_activity` →
  `setattr` on the **screen**. That is the write-back proof for the setter
  binding;
- resumes and asserts exactly one reconciliation ran and the latch cleared.

### 9.5 — `./scripts/preflight.sh`

**All green**, six checks:

```
=== generated stylesheets ===        all 10 bundles reproduce
=== profile-owned path census ===    48 occurrences / 18 files, all matched
=== production diagnostic inventory ===
                                     575 owners, 1336 TASK-492 calls,
                                     7610 TASK-494 calls, 11 sink files
=== backlog task ids ===             no duplicates across 3306 task files
=== chachanotes table allowlist ===  105 declared tables, all present
=== index plan pins ===              270 indexes / 270 census rows / 57 pinned
preflight: all derived-artifact checks passed.
```

No drift reported, so `check_persistent_diagnostic_inventory.py --write` was
never run.

Per the brief's ruling, **no third full paired-baseline xdist sweep** was run
this round.

---

## 10. `test_library_entry_compose_once.py` — 3 failures, paired against dev

These were the battery's only unexplained failures, so they got the full
recipe §7 treatment rather than an assertion.

**Cheap structural checks first.** The file is byte-identical to
`origin/dev` (`git diff --quiet origin/dev -- <file>` is clean) and contains
**zero** occurrences of `ingest`. This session's only edit to the helper
module it imports from (`test_library_shell.py`) is purely additive inside
`wire_bypass_ingest_controller` — verified by `git diff` on the unstaged
hunks, which touch no existing line.

**Failure mode is Notes-surface DOM**, not ingest:

```
E   AssertionError: #library-notes-row-0 never mounted within 30.0s (1180 polls).
    Visible text: … ▸ Notes (2) … Unfiled   Q3 retro   Reading list …
    Tests/UI/test_library_shell.py:3696
```

The notes exist and render; the flat-row selector the helper waits for never
appears.

**Paired baseline run.** A throwaway `git worktree add --detach` at
`origin/dev` (`2c9c14418`) was placed under the session scratchpad — never an
in-place checkout — with its **own** `uv venv` + `uv pip install -e ".[dev]"`.
Import provenance was verified before trusting any result, and the first
check caught the classic trap (running the devbase interpreter from the
branch's cwd resolves to the *branch's* package):

```
$ ./.venv/bin/python -c "import tldw_chatbook; print(tldw_chatbook.__file__)"   # wrong cwd
/Users/…/.worktrees/library-decomp-foundation/tldw_chatbook/__init__.py     <- BRANCH, not devbase

$ cd <scratchpad>/devbase && ./.venv/bin/python -c "…"                          # correct
<scratchpad>/devbase/tldw_chatbook/__init__.py
```

Identical invocation on both sides
(`-m pytest Tests/UI/test_library_entry_compose_once.py -p no:randomly -q`):

| Side | Result |
|---|---|
| branch (merged tree) | **3 failed, 85 passed** (154.51s) |
| `origin/dev` baseline | **4 failed, 84 passed** (167.32s) |

Failure-name diff:

| Name | branch | `origin/dev` |
|---|---|---|
| `test_library_graduation_announcement_survives_reconcile_and_same_route_replace` | FAIL | **FAIL** |
| `test_library_notes_recompose_does_not_steal_newer_focus[reconcile]` | FAIL | **FAIL** |
| `test_library_notes_recompose_does_not_steal_newer_focus[replace]` | FAIL | **FAIL** |
| `test_source_worker_completion_during_resume_dispatch_reconciles_once` | **pass** | FAIL |

**Zero branch-unique failures.** All three reproduce identically on
`origin/dev`, and the branch additionally **passes** a fourth test that dev
fails. The throwaway worktree was removed afterwards.

---

## 11. Concerns

1. **Three pre-existing `origin/dev` ratchet reds, carried forward
   unchanged.** `chat_screen.py`'s budget,
   `test_task_22507_4_does_not_worsen_chat_screen_base`, and
   `library_media_browse_controller.py`. For each, both the measured file
   **and** the pin are byte-identical to `origin/dev`, so this branch cannot
   have caused them. Pins were deliberately **not** raised: absorbing dev's
   un-owned creep into this merge would silence guards that are correctly
   firing at dev. Same disposition as the previous round; still needs an
   owner on dev (recipe §7, backlog task-31249).

2. **`test_library_entry_compose_once.py` is 3/88 red on BOTH trees.** A
   Notes-surface DOM failure (`#library-notes-row-0` never mounts) in a file
   byte-identical to dev's with zero ingest references. Not caused by this
   merge — proven by the paired run in §10 — but it does mean a library test
   file ships red on dev. Flagging rather than fixing: out of this brief's
   scope, and dev owns it.

3. **A dev-red test that this branch turns GREEN, mechanism unconfirmed.**
   `test_source_worker_completion_during_resume_dispatch_reconciles_once`
   fails on `origin/dev` and passes on the merged branch. The plausible
   mechanism is §5's `on_screen_suspend` fix: on dev, the ingest
   path-debounce timer is never actually stopped at suspend (the string-loop
   `getattr` no-ops for a *different* reason there — the field is flat on dev
   — but this branch's explicit state-object stop is the only behavioural
   difference in that hook), and a stray timer firing during resume dispatch
   is exactly the shape of a double reconcile. **I did not confirm this
   causally** — it could equally be ordering noise in a file where three
   other tests are already failing on both sides. Recording it as an
   observation, not a claim, and not counting it as a win.

4. **The interim accessor boundary is now TWO fields wide, not one.**
   `_library_ingest_analyze_outcomes` and `_library_ingest_suspended_
   activity` are both ingest-exclusive fields the controller reaches through
   bespoke bindings rather than through `LibraryIngestState` — the ingest
   state boundary is 20 fields where it should be 22. Deliberate (minimum-
   risk way to keep dev's edits byte-for-byte inside a merge commit),
   time-boxed by task-31651 (ACs #9/#10 added this round), and
   cross-referenced from the controller's module docstring so it cannot
   quietly become permanent. The third new accessor,
   `library_screen_suspended_accessor`, is **not** in that debt: it is
   screen-wide lifecycle state and an accessor is its correct permanent
   shape.

5. **The brief's arity figure was low and is corrected.** The brief said
   "38 → 40 keyword-only"; measured reality is **38 → 41** (43 total),
   because the getter/setter pair is two parameters under the established
   pattern, not one. The brief explicitly instructed measurement over
   assumption, so this is the instruction working, not a contradiction. The
   measured figure is what appears in the ratchet comment and the merge
   commit message.

6. **Deviations from the brief, each deliberate and small:**
   - The brief's census scope (20 state fields + analyze_outcomes) does not
     cover dev's 4 **new** fields, so those were swept separately for
     bypass-screen uses. Result: only this session's own additions. No extra
     seeding needed.
   - No third full paired-baseline xdist sweep was run — the brief ruled it
     out. A **targeted** paired run was still done for the one file with
     unexplained failures (§10), since a §7 step-4 comparison on one file is
     cheap and was the difference between evidence and assertion.
   - Everything landed in the single merge commit, including the task-31651
     edit and the bypass seeds. They are inseparable from the merge (the
     seeds are required for the merged tree's tests to pass), so splitting
     them into a follow-up would have produced a commit that does not build
     green on its own.
   - This report is committed as a second, documentation-only commit, per
     the series' precedent (`.superpowers/` is `.gitignore`d and every other
     report here is force-added and tracked).

7. **Not verified live.** This is a merge plus a doc pass; no UI behaviour
   was exercised by hand. The port's behaviour is covered by dev's own
   `test_suspended_library_gates_ingest_dom_work_until_resume` (§9.4), which
   drives a real `Pilot` through the screen delegator into the ported
   controller body and asserts the latch's write-back on the screen.
