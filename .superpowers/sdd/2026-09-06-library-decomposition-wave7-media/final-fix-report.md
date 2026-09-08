# Wave-7 media — final review conditions + the program's largest dev reconciliation

**Status: COMPLETE.** All five merge-side conditions executed. The
`origin/dev` reconciliation landed as a true merge commit with a
census-driven semantic port; the exclusion-debt owner is filed.

| | |
|---|---|
| Merge commit | `3c11f902a` (parents `17ba6d34f`, `0bb00beaf`) |
| Follow-up commit | `2c4c4d709` |
| Merge-base | `761416317` |
| dev commits merged | **306** (the review measured 292; re-measured at merge time) |
| Working tree | clean except `progress.md`, modified at session start and deliberately never staged — verified absent from both commits |

---

## 0. Drift, re-measured

```
git log --oneline HEAD..origin/dev | wc -l   ->  306
git merge-base HEAD origin/dev               ->  7614163173ba009701725d757984ceed52404056
origin/dev tip                                ->  0bb00beaf
```

The review's three headline quantities all reproduced, derived fresh by AST
rather than carried:

| Review's figure | Measured at merge time |
|---|---|
| 11 mover bodies edited on dev | **11**, exactly (net +63 lines) |
| ~49 new dev screen methods, 16+ media-named | **29** new method NAMES, **16** media-named |
| zero dev callers of a pruned delegator | **zero** in the unbound spelling — but see §5, which found six in a spelling nobody had censused |

The 49-vs-29 gap is an instrument difference and is recorded rather than
reconciled: 29 is the count of names present on `origin/dev`'s `LibraryScreen`
and absent at the merge-base, measured by AST set difference. Both figures
stand; only 29 is reproducible from the trees.

---

## 1. Conflicts — per-conflict record

`git merge origin/dev` conflicted in **4 files / 18 hunks**. The screen file
was re-checked out with `--conflict=diff3` so every hunk could be read against
its BASE; resolving from `ours`/`theirs` alone would not have distinguished
"dev edited a body" from "dev deleted a body" from "dev added a method beside
one".

### `tldw_chatbook/UI/Screens/library_screen.py` — 15 hunks

| # | Enclosing | base/ours/theirs | Resolution |
|---|---|---|---|
| 1 | `__init__` | 24/0/29 | OURS + dev's 5 genuinely-new `_library_media_rename_cache` lines |
| 2 | `_apply_navigation_context_state` | 144/144/0 | THEIRS (dev moved the body out) + a port into dev's module |
| 3 | `_apply_library_media_active_surface` … `_finish_library_media_list_return` | 66/2/77 | OURS + dev's new `_sync_library_media_surfaces_or_recompose`; body edit ported |
| 4 | `_build_library_media_state` | 46/1/48 | OURS; body edit ported |
| 5 | `_library_media_canvas_presentation` | 22/1/43 | OURS; body edit ported |
| 6 | `_select_library_media_reader_row` | 58/2/57 | OURS; body edit ported |
| 7 | `_library_media_trash_canvas_presentation` | 37/1/30 | OURS; body edit ported |
| 8 | `_build_library_media_reader` | 125/1/136 | OURS; body edit ported |
| 9 | `_build_library_media_viewer_display_state` | 13/0/50 | OURS (stays pruned) + dev's new `_library_media_speaker_rename_facts` |
| 10 | `_sync_library_media_viewer_state` | 197/1/220 | OURS; body edit ported |
| 11 | `handle_library_media_reader_more` | 7/1/20 | OURS; body edit ported |
| 12 | `handle_library_media_reader_mode` | 26/1/35 | OURS; body edit ported |
| 13 | `handle_library_media_reader_find` | 4/4/5 | **COMBINED** — both sides edited |
| 14 | `_restore_library_media_focus` | 76/1/79 | OURS; docstring edit ported |
| 15 | `_sync_library_media_viewer_or_recompose` | 46/1/146 | OURS + dev's two new methods |

**#1 in detail.** Dev's side of this hunk was 29 lines, of which 24 were the
seven analyze fields this wave moved to `LibraryMediaState` and their comment.
The remaining **5** — comment plus declaration — are
`_library_media_rename_cache`, a field dev created inside the range
(`git grep` at the merge-base: absent; on dev: 3 uses, all in
`library_screen.py`). It is dev's, it is not one of the 82, and it stays flat
on the screen. Its own comment said "Same motivation as the reason cache
above", and the reason cache is no longer above it; a three-line note records
where the referent went rather than silently leaving a dangling reference.

**#13 in detail — the only genuinely two-sided hunk.**
`handle_library_media_reader_find` is screen-resident (not a mover), and both
sides changed the same four lines: this branch retargeted the two field
writes, dev replaced the two-call tail with its new
`_after_library_media_viewer_sync` seam. Taking either side whole would have
lost the other's work silently — a retarget lost means two stray flat
attributes; dev's edit lost means the Find focus race dev fixed comes back.
Resolved by hand as the union:

```python
        self._media_state.find_open = True
        self._media_state.find_focus_pending = True
        self._after_library_media_viewer_sync(
            self._focus_library_media_content_search_input
        )
```

### `Tests/UI/test_library_multiselect_media.py` — 1 hunk

base 1 / ours 1 / theirs 134. Ours retargeted one assertion; dev appended a
133-line task-28009 test block at the same offset. **Kept both.** This file
is the ADR-055 guard's home; the interlock check is verified intact in §7.

### `Tests/UI/test_library_shell.py` — 2 hunks

- #1: dev changed a status string (`"Match 1 of 2 matches"` → `"Match 1 of 2"`);
  ours retargeted the two asserts beneath it. **Kept both** (dev's line 1,
  ours' lines 2-3).
- #2: ours retargeted two asserts; dev appended a 166-line task-31946 block.
  **Kept both.**

### `Docs/security/production-diagnostic-inventory.json`

`git checkout --theirs`, then the drift rows were **read before** regenerating:

```
owner_files: 589 -> 590
  + library_media_controller.py [TASK-494] count=7
  ~ library_screen.py 95 -> 88  (-7 diagnostic call(s))
```

`--statements … --since origin/dev` recovered the statement text the aggregate
digest cannot show. All seven move with **identical digests** —
`da902fd5e6dcc95b`, `e769cfc2bcc5ada0`, `b3630f7f9b9cd752`,
`762b411a7afcd2c2`, `d493cee97e27ff80`, `b841df0b6be9e298`,
`f9ac0ed52a75ab56` — removed from the screen, added to the controller,
byte-for-byte: no rewording, no re-levelling, no new interpolation of user
content, secrets or paths. Only then `--write`. Re-verified: **590 owners,
1347 TASK-492, 53 TASK-31551, 7673 TASK-494, 12 sink files.**

---

## 2. The ports — verification transcript

### Precondition, checked before writing anything

Every one of the 11 controller bodies was still **byte-identical to the
merge-base screen body**:

```
IDENTICAL  _apply_library_media_active_surface        (controller @1620-1649)
IDENTICAL  _build_library_media_reader                (controller @3493-3555)
IDENTICAL  _build_library_media_state                 (controller @1774-1820)
IDENTICAL  _library_media_canvas_presentation         (controller @1887-1909)
IDENTICAL  _library_media_trash_canvas_presentation   (controller @2161-2198)
IDENTICAL  _restore_library_media_focus               (controller @3948-4024)
IDENTICAL  _select_library_media_reader_row           (controller @2024-2081)
IDENTICAL  _sync_library_media_viewer_or_recompose    (controller @4026-4072)
IDENTICAL  _sync_library_media_viewer_state           (controller @3633-3830)
IDENTICAL  handle_library_media_reader_mode           (controller @3891-3917)
IDENTICAL  handle_library_media_reader_more           (controller @3846-3853)
```

That is what makes the port a **wholesale replacement** with dev's version
rather than a hand-applied patch: exact by construction, not by care.

### Result

```
=== PORT IDENTITY (text + AST) ===
  text=IDENTICAL ast=IDENTICAL  _apply_library_media_active_surface
  text=IDENTICAL ast=IDENTICAL  _build_library_media_reader
  text=IDENTICAL ast=IDENTICAL  _build_library_media_state
  text=IDENTICAL ast=IDENTICAL  _library_media_canvas_presentation
  text=IDENTICAL ast=IDENTICAL  _library_media_trash_canvas_presentation
  text=IDENTICAL ast=IDENTICAL  _restore_library_media_focus
  text=IDENTICAL ast=IDENTICAL  _select_library_media_reader_row
  text=IDENTICAL ast=IDENTICAL  _sync_library_media_viewer_or_recompose
  text=IDENTICAL ast=IDENTICAL  _sync_library_media_viewer_state
  text=IDENTICAL ast=IDENTICAL  handle_library_media_reader_mode
  text=IDENTICAL ast=IDENTICAL  handle_library_media_reader_more
ALL 11 IDENTICAL: True
```

**Zero substitutions.** The review predicted "most ports should come out
byte-identical"; the answer is all eleven, and the reason is structural: the
controller exposes every state field under its original flat name through the
generated shim block, and every other name through a constructor dependency
under the same name.

Per-body line deltas (dev's own, which the controller now carries):
`_sync_library_media_viewer_state` +23, `_library_media_canvas_presentation`
+21, `handle_library_media_reader_more` +13, `_build_library_media_reader`
+11, `handle_library_media_reader_mode` +9, `_restore_library_media_focus`
+3, `_build_library_media_state` +2, `_select_library_media_reader_row` −1,
`_sync_library_media_viewer_or_recompose` −2,
`_library_media_trash_canvas_presentation` −7,
`_apply_library_media_active_surface` −9. **Net +63.**

### The eight new bindings, derived not guessed

An AST walk over the whole controller class collected every `self.<attr>` and
resolved each against the class's own definitions, `__init__` assignments and
the 82 generated shims. It returned 33 unresolved; 25 were shim-covered state
fields, leaving **exactly 8** genuinely unbound — all of them dev's new screen
methods, reached by the ported bodies:

```
_after_library_media_viewer_sync              <- handle_library_media_reader_mode, handle_library_media_reader_more
_decorate_library_media_reviewed              <- _build_library_media_state
_library_media_can_rename_speakers            <- _library_media_canvas_presentation
_library_media_list_unselectable              <- _build_library_media_reader, _library_media_canvas_presentation
_library_media_selected_backing_id            <- _build_library_media_reader, _library_media_canvas_presentation, _sync_library_media_viewer_state
_library_media_trash_action_disabled_reason   <- _library_media_trash_canvas_presentation
_queue_after_library_media_viewer_recompose   <- _sync_library_media_viewer_state
_sync_library_media_surfaces_or_recompose     <- _apply_library_media_active_surface, _sync_library_media_viewer_or_recompose
```

Each added as a group-(e) named late-binding callable (constructor parameter →
`self._<name>_fn` → `@property`), the shape the other 35 already use, and
wired at the construction site as `lambda *a, **k: self._<name>(*a, **k)`.
They stay **screen-resident**: they are dev's new methods, not this wave's
movers, and a late-binding lambda re-reads them at call time.

Keyword-only arity **83 → 91**; construction-site kwargs **83 → 91**
(measured with `ast`, both sides agreeing). Post-change AST re-run:
`UNBOUND (excluding generated state shims): []`.

### The port into dev's new module

Conflict #2's resolution took dev's side — dev moved
`_apply_navigation_context_state`'s whole 144-line body into its new
`library_unavailable_navigation.py`. The branch's two retargets lived inside
that body, so they had to follow it:

```diff
         if open_source_type == "media":
-            self._selected_media_id = open_source_id
-            self._library_media_view = "list"
+            self._media_state.selected_media_id = open_source_id
+            self._media_state.view = "list"
```

Both are **WRITES**. Left flat they would not have raised: they would have
created two stray attributes split from the state object, and every "open in
Library" media deep link would have painted the wrong surface with no
exception and no red test. This is the wave-6 precedent's exact shape (dev's
new controller carrying a flat reference), one severity level worse because
wave-6's was a read.

---

## 3. Census over the merged tree — all five spellings

The 82 field names were derived programmatically from `LibraryMediaState`'s
dataclass fields through `media_state_shim_attr()` — the single authoritative
mapping — never hand-listed. `FIELD COUNT: 82`.

### Spellings 1–4 (attribute / quoted-string / bare-assignment / patch-target table)

One whole-word regex covers all four at once, over **every file** (not only
`.py`) under `tldw_chatbook/` and `Tests/`. First pass: **475 hits / 24 `.py`
files / ZERO non-`.py`**.

Dispositions:

| Class | Count | Disposition |
|---|---|---|
| Controller's own shims and moved bodies | 339 | Correct — the file that owns them |
| `library_media_state.py` prose | 8 | Correct — defines them |
| Prose / comments / docstrings elsewhere | ~90 | Left; 2 stale dev references fixed (below) |
| Wiring test's mapping assertions | 5 | Correct — tests the function, not an attribute |
| **LIVE screen-side references dev added** | **10** | 9 RETARGETED, 1 ruled not-a-hit |
| **LIVE test-side `screen.<flat>`** | **38** | RETARGETED across 7 files |
| **Fixture-shape fakes** | 3 sites | RESEEDED with `_media_state` |

### The 10 live screen-side references, individually

| Line | Method | Shape | Disposition |
|---|---|---|---|
| 3204 | `__init__` | `getattr(app_instance, "_library_media_layout_refresh_generation", 0)` | **LEFT** — see below |
| 3960 | `_library_route_shortcuts_for_current_state` | read | `_media_state.view` |
| 6438 | `_handle_library_media_speaker_renamed` | read | `_media_state.detail` |
| 6447 | `_handle_library_media_speaker_renamed` | **WRITE** | `_media_state.detail` |
| 8916 | `_library_focus_channel_owns_this_window` | read | `_media_state.find_focus_pending` |
| 15410 | `_library_media_selected_backing_id` | read | `_media_state.selected_media_id` |
| 15935 | `_library_media_trash_action_disabled_reason` | read | `_media_state.bulk_delete_in_flight` |
| 15948 | `_library_media_trash_actions_live` | **`getattr` with default** | `getattr(self._media_state, "view", _MEDIA_VIEW_LIST)` |
| 31282 | `_review_cursor_for_display` | read | `_media_state.reader_session` |
| 32800 | `_library_media_speaker_rename_facts` | read | `_media_state.reader_session` |

Plus the two writes in `library_unavailable_navigation.py` (§2) — **three
writes in total**, the silent class.

**Line 3204 is the one ruled NOT a hit, and it was checked rather than
assumed.** It reads `_library_media_layout_refresh_generation` off
`app_instance`, a different object from the screen, as the legacy-name
fallback in a rename chain: the live writer is
`settings_screen.py:28291`, which sets
`app_instance._library_reader_layout_refresh_generation`. Nothing anywhere
sets the media-named form on the app. The line is present and untouched on the
**pre-merge branch tip** (`17ba6d34f:3127`), so the branch's own cleanup
already classified it this way; the merge did not revisit that call, only
verified it.

**Line 15948 kept dev's `getattr` SHAPE.** The mechanical answer is
`self._media_state.view`, and that is what the first pass produced. It was
changed back to a `getattr` with the same default after noticing this file's
own retargeted sibling at `:25009` uses
`getattr(self._media_state, "view", _MEDIA_VIEW_LIST)`: the default is what
keeps a fixture-built `_media_state` double working, and dropping it would
have been a behaviour change smuggled in as a receiver swap.

### Two stale dev prose references

Found by a differential prose sweep — every line mentioning a deleted flat
name that exists now but did not exist on the pre-merge branch tip. Eight
lines; six were this merge's own retarget comments (which name the old
spelling on purpose). The two real ones, both dev's, both fixed:

- `library_screen.py:15404` — a docstring saying ``self._selected_media_id``
  is the canonical id form.
- `test_library_media_render_fixes.py:3005` — a docstring naming
  ``_library_media_progress_restored_id``.

This is recipe §3's "deleted FIELD names' own prose sweep", which exists
because that class is invisible to every automated guard in the battery.

### Spelling 5 — the computed name

`ast.JoinedStr` plus `%`/`+`/`.format()` string building, over both roots,
keeping any node whose literal fragments could compose one of the 82 in order.
Three candidates:

| Site | Verdict |
|---|---|
| `canvas_sync.py:227` — `f"_library_{kind}_row_selection"` | **REAL, and already correct.** The branch's dotted media branch (`"_media_state.row_selection" if kind == "media"`) survived dev's own hunk to this file intact — read live, not assumed — so the f-string is unreachable for `kind == "media"` |
| `library_media_state.py:254` — `"_library_media_" + field_name` | The mapping's own definition |
| `ccp_validation_decorators.py:271` — `f"selected_{item_type}_id"` | **False positive.** No leading underscore, so it cannot compose `_selected_media_id`; and it is a CCP screen's decorator, a different object entirely |

A first, looser matcher returned 44 candidates on single generic fragments
like `media_`. Tightening to ≥55% literal coverage and ≥2 fragments (or one
≥12 chars) reduced it to 3. Both runs are recorded because the loose one is
the honest starting point and the tight one is the answer.

### Final state

```
=== 82 DELETED FLAT FIELD NAMES: 418 hits, 21 .py files, non-.py: 0 ===
    339  tldw_chatbook/UI/Library_Modules/library_media_controller.py   (owner)
     31  tldw_chatbook/UI/Screens/library_screen.py                     (prose)
      8  tldw_chatbook/UI/Library_Modules/library_media_state.py        (defines)
      … all remaining files: prose, or the wiring test's mapping assertions
```

**Zero live readers or writers of any deleted flat name outside the two files
that legitimately own one.**

---

## 4. The 22 pruned delegators — and the census gap this found

The review measured zero dev callers of a pruned name; the unbound spelling
reproduces that at merge time:

```
=== LibraryScreen.<MOVER> attribute references in Tests/ === 4 total
  test_library_media_reader_flow.py:1303,1337  handle_library_media_reader_more
  test_library_media_trash.py:3459,3481        _apply_library_media_list_return
=== references to any of the 22 PRUNED names, unbound spelling === ZERO
```

**That census is incomplete, and the merge proved it.** It searches
`LibraryScreen.<name>` (unbound) and the bare-quoted-string spelling. Neither
sees `screen.<name>(...)` on a real instance. Re-run over ALL spellings:

```
=== 22 PRUNED DELEGATOR NAMES: 117 hits, 5 .py files ===
     63  library_media_controller.py                    (owner)
     44  test_library_media_wiring.py                   (the pin)
      6  test_library_media_viewer_speaker_rename.py    <- LIVE, dev's new file
      2  test_local_media_reading_service.py            (prose)
      1  library_media_state.py                         (prose)
```

Six bound calls to `_build_library_media_viewer_display_state` on a real
screen, in a file dev created. They failed with a plain `AttributeError` the
moment the file ran — caught by the battery, not by the census that should
have caught it. The gap is written into TASK-31976 so the next series' census
includes the bound-receiver spelling.

### Two rulings, both recorded with their rejected alternative

**(a) `_build_library_media_viewer_display_state` — retarget the test, do not
restore the delegator.** The 6 calls now read
`screen._media_controller._build_library_media_viewer_display_state(...)`.
Restoring the pruned delegator was rejected on the prune census's own terms:
its only caller would be a test, and a delegator kept alive by a test is
exactly what the prune removed. Result: **10 passed** in that file.

**(b) `handle_library_media_reader_more` — retarget the fixture, do not revert
the mover.** Dev added the wave's first unbound-fake-self fixture for a name
this wave had already MOVED (2 call sites). Recipe §3's opening rule, read
literally, says a fixtured name stays screen-routed — i.e. revert the mover.
That was rejected here because it would re-shape the wave's pinned 140-mover
set, the controller docstring's exclusion arithmetic and both ratchets inside
a merge commit, for a test that arrived after the cleanup census that governs
the set. §3 itself lists "retarget the fixture to match" as a sanctioned fix
for this shape, and the cleanup PR applied it to 37 other fakes.

A ~30-line `_MoreHandlerControllerFake` in the test file reproduces the one
production hop — the generated state shim and the group-(e) property — so the
**real controller body still runs** against the file's own screen fake. The
retargeted fixture no longer names `LibraryScreen.<mover>` at all, so the
contradiction with §3's census is removed rather than papered over. Result:
**3 passed**.

Both rulings, and both rejected alternatives, are written into TASK-31976.

---

## 5. Pins — fresh `_measure()` on the merged tree

Each measured with the owning file's **own** `_measure`, on the merged tree,
immediately before writing.

| Row | Before | After |
|---|---|---|
| `test_screen_size_ratchet.py` — `LibraryScreen` | `(34669, 1260)` | **`(35626, 1289)`** |
| `test_library_modules_size_ratchet.py` — `library_media_controller.py` | `4496` | **`4612`** |
| `test_library_modules_size_ratchet.py` — `library_unavailable_navigation.py` | `811` (dev's) | **`817`** |
| `test_library_modules_size_ratchet.py` — `library_media_browse_controller.py` | `371` | **`371`, unchanged, standing red** |

### The method count is a SET check

```
merge-base: 1269   dev: 1298   branch: 1247   merged: 1276
dev NEW (dev - base): 29
merged - branch: 29   == dev-new?  True
branch - merged (LOST delegators): NONE
extras in merged not from dev:     NONE
```

`merged − branch` is **exactly** dev's 29 new names; `branch − merged` is
**empty**. That pair is what proves the resolution neither re-added a moved
body nor dropped a wave-7 delegator — the two failure modes a naive resolution
produces. (These are unique-NAME counts; the ratchet's `_measure` counts
`FunctionDef` nodes, so it reads 1260 → 1289. Same delta, +29.)

### The line delta reconciles against dev's, not against an estimate

```
base 37537   dev 38525   branch 34669   merged 35626
dev's own delta over the range: +988
merged - branch:                +957
```

988 − 63 (the ported mover edits, which the screen does not take) = **925**
from dev. This merge's own **+32**: +24 for the eight construction-site kwargs
(3 lines each), +5 for the `getattr`-shape comment at `:15948`, +3 for the
`_library_media_rename_cache` note. **925 + 24 + 5 + 3 = 957.** Exact.

### Controller +116, itemised from the diff hunks

+63 ported bodies, +8 signature parameters, +8 constructor assignments, +37
properties and the block's new comment. `162 − 46 = 116`. Exact.

### `library_media_browse_controller` — refused for the fourth wave

| Wave | Measured | Pin |
|---|---|---|
| 6 | 410 | 371 |
| **7** | **649** | **371** |

The review predicted ~602; the measured value is 649. Red on `origin/dev`
itself, verified in the isolated baseline. No Library extraction has ever
touched the file. Re-pinning it from a passing branch would launder dev-side
debt behind a Library merge, which the ratchet's own guidance forbids. The row
now carries a comment stating both numbers and the refusal; recipe §7 records
that the overshoot **more than tripled in one wave while the pin sat still**,
which is the part worth saying out loud about leaving a red for its owner.

A second, mirror-image dev row appeared this wave and is treated identically:
`test_budget_is_not_left_slack_after_a_move[library_conversations_controller.py]`
— dev shrank that file 1738 → 1686 without lowering its row (52 slack against
a 50 tolerance). Dev's move, dev lowers the row.

---

## 6. Battery

Baseline throughout: an isolated `origin/dev` worktree with its **own** venv.
Per recipe §3's rule added by this wave's own close, parity was **proven
before any count was read**:

```
python -V                      branch 3.14.2    baseline 3.14.2
uv pip list | names | sort     IDENTICAL PACKAGE SETS (106 each)
import tldw_chatbook.__file__  each tree resolves ITS OWN
```

The tree-resolution check earned its place immediately: run from the wrong
cwd, the baseline interpreter resolved the *branch's* package (the editable-
finder trap). Re-run from the worktree root it resolves its own.

| Suite | Merged branch | `origin/dev` baseline |
|---|---|---|
| 8 library wiring suites | **59 passed, 0 failed** | — |
| support-layer + preimport-closure + media characterization + screen-reuse + modal-dismissal + selection-updates | 190 passed, 3 failed | — |
| both size ratchets | 42 passed, 4 failed | — |
| *(matched batch of the above two, 5 files)* | **7 failed / 220 passed** | **8 failed / 216 passed** |
| recompose ratchet + `ui_ready` preimport census | 24 passed, 2 failed | 21 passed, 2 failed (same names) |
| dev's 18 new/changed media test files | 936 passed, 7 failed → **0 branch-unique after the §4 fix** | 939 passed, 4 failed |
| `Tests/UI/test_screen_navigation.py` | 110 passed, **32 failed** | 110 passed, **32 failed — same 32 names** |
| ADR-055 one-flag interlock + canvas_sync dotted-path guard | **2 passed, 0 failed** | — |
| `./scripts/preflight.sh` | **all seven checks passed** | — |

### Every red, proven

**Matched batch (ratchets + reuse + modal + selection): 7 shared, ZERO
branch-unique, 1 baseline-unique.** The baseline-unique one is dev's own
`library_screen.py` ratchet row, which this merge's re-pin turns **green** —
the merge fixes a dev red rather than adding one.

The seven shared:

1. `chat_screen.py` budget — documented in recipe §7 since wave 5.
2. `test_task_22507_4_does_not_worsen_chat_screen_base` — same.
3. `library_media_browse_controller.py` — §5 above.
4. `library_conversations_controller.py` slack — §5 above. New this wave.
5. `test_on_screen_suspend_stops_every_timer_in_isolation` — **dev's own
   regression, and a nice irony**: dev added
   `self._unavailable_navigation.clear_character_return(self)` to
   `on_screen_suspend` without seeding that attribute in the test's
   `LibraryScreen.__new__` fixture — the same `__new__`-bypass shape every
   state PR in this program has had to seed, now bitten from the dev side.
   `git diff` confirms dev never touched that test file.
6. `test_library_modal_inventory_matches_declared_edges_bidirectionally` —
   the inventory's AST resolver cannot resolve dev's
   `SkillImportChoiceModal(snapshot.candidates)`. Not media.
7. `test_tier1_toggle_falls_back_to_recompose_on_query_one_failure` —
   documented in §7 by wave-7 task 3's own fix round.

**The recompose ratchet was checked on assertion TEXT, not just on name.**
A same-name failure with different numbers would hide a regression:

```
MERGED:   66 found, 63 allowed
BASELINE: 66 found, 63 allowed
```

**`test_screen_navigation.py` deserves its own note.** Wave 6 recorded ~30
failures here with churning membership and called the file flaky. At wave 7
the count is 32 and the failing NAME SET is **identical** on both trees. That
is a different and worse diagnosis than flake: a standing, unowned dev-side
regression across 32 navigation tests. Filed on TASK-31249 and recorded in
recipe §7 so no future wave reads a green expectation into it.

### Paired post-merge spot sweep — the 6 heaviest media files

`test_library_media_trash.py` (4846), `test_library_media_render_fixes.py`
(3727), `test_library_multiselect_media.py` (3310),
`test_library_media_reader_flow.py` (2828),
`test_library_media_return_settlement.py` (2493),
`test_library_media_browse_controller.py` (1010).

| | failed | passed |
|---|---|---|
| merged branch | **3** | 483 |
| `origin/dev` baseline | 4 | 482 |

**3 shared, ZERO branch-unique, 1 baseline-unique.** The shared three are the
`test_media_trash_focus_initial_*` / `_retry_focus_intent` group; the
baseline-unique one
(`test_trash_back_exact_scroll_precedes_captured_control_focus`) passes on the
branch. Both sides ran 486 tests.

---

## 7. Guard integrity spot-checks

- **ADR-055 one-flag interlock.** Its home file
  (`test_library_multiselect_media.py`) took a 133-line dev addition through a
  conflict; the guard passes on the merged tree, including its two regex
  counts (`_media_state.bulk_delete_in_flight\s*=\s*True` exactly once;
  `group=['"]library_media_bulk_delete['"]` exactly once) and its six-handler
  `inspect.getsource` sweep. The census's own edit to that file (one assertion
  retarget) does not touch a claim site.
- **canvas_sync fifth-spelling guard.**
  `test_media_row_toggle_resolves_the_dotted_state_path` passes, and the
  production dotted branch it guards was read live after the merge — dev's own
  hunk to `canvas_sync.py` did not disturb it.
- **Preimport closure / `ui_ready` census.** The controller stays off the
  registry's import-only walk; both pass, so 306 dev commits did not make it
  eager.

---

## 8. Exclusion-debt owner — TASK-31976, filed not appended

The condition said to extend dev's Library UI test-debt census
(**TASK-31249**, which `origin/dev` commit `880867297` extends), *or* file a
new task if extending is wrong for that task's shape.

**Extending was wrong, and the reason is in the task's acceptance criteria.**
TASK-31249 is a census of Library UI tests that FAIL on clean dev; its ACs are
"each of the six tests passes on dev", "these four files run green in separate
processes". The media exclusion debt is about fixture SHAPES blocking further
extraction while every one of those tests passes. Different fix, different
owner, different done condition. Appending would have made both harder to
close.

**TASK-31976** (id swept across every remote ref and every local backlog
directory — observed maximum 31975, then re-verified unique after creation;
preflight's duplicate-id check green at 3420 task files). It carries:

- the 89/22 split and where each number is re-derivable from, with an explicit
  instruction to re-derive rather than carry;
- six outcome-shaped ACs, including that the 9/7 monkeypatch split be
  re-derived at the time of the work;
- the 7 named zero-mover-caller candidates;
- §3's governing rule stated as governing — draining this debt means
  retargeting fixtures FIRST, not re-arguing exclusions against a narrower
  test;
- the two live instances this merge absorbed (§4), each with its rejected
  alternative;
- the bound-receiver census gap.

TASK-31249 was still extended, with the five wave-7 `origin/dev` reds and a
pointer to TASK-31976, so the two are discoverable from each other. Recipe §7
carries the same reds where future waves actually read them.

---

## 9. Concerns

1. **`library_media_browse_controller.py` is red and stays red, fourth wave
   running — and it is now 649 vs 371.** 410 at wave 6. The refusal is right,
   but the trend is the story: nobody on dev has picked it up across four
   waves, and the file has gained 239 lines in the interval. It needs a named
   owner, not another wave's paragraph.

2. **`test_screen_navigation.py`: 32 failed / 110 passed, same names on both
   trees.** Wave 6 called this file flaky. It is not flaky now; it is broken,
   and the breakage is unowned dev-side. That is 22% of a navigation suite
   that every Library wave runs as a smoke test and every Library wave has
   discounted.

3. **Ruling (b) in §4 is a judgement call and could reasonably have gone the
   other way.** Recipe §3's opening rule, read literally, says
   `handle_library_media_reader_more` should have been reverted to the screen
   because a test now fixtures it. I retargeted the fixture instead, on the
   grounds that the test post-dates the census that governs the mover set and
   that re-shaping a pinned architectural fact inside a merge commit is worse.
   A reviewer who disagrees should be able to act on it cheaply: the change is
   confined to one test file and one ~30-line class. Both sides of the
   argument are in TASK-31976 and in the merge message.

4. **Dev is now adding fixtures and callers against a screen surface this
   program has moved.** Two instances in 306 commits (one moved name, one
   pruned name), both absorbed by hand here. Neither was caught by a census —
   one because no census covered the spelling, one because the shape is
   legitimate on dev and only conflicts with the branch. As long as this
   branch stays unlanded the rate is likely to hold, and the next
   reconciliation should expect the same class rather than assume the
   mitigations transfer.

5. **`library_unavailable_navigation.py` now carries a wave-7-specific
   comment**, exactly as `library_navigation_controller.py` came out of
   wave 6. If dev edits that region concurrently, the six-line comment is the
   likely conflict site — noted so the next reconciliation recognises it.

6. **The 49-vs-29 new-method count is not reconciled.** The review's figure and
   this merge's AST measurement disagree, and the difference is
   instrument-vs-instrument (as with wave-7's own 1,214/1,204 census pair).
   Both are recorded; only 29 is reproducible from the trees.

---

# Round 2 — the 75-commit `fix/media-riders` race

**Status: COMPLETE.** `origin/dev` advanced 75 more commits while PR #2501
was open — the `fix/media-riders` series, actively editing the same media
cluster round 1 had just reconciled — and the PR went CONFLICTING. Same
procedure, smaller scale.

| | |
|---|---|
| Merge commit | `16c45b545` (parents `3a317a391`, `37bf45fb6`) |
| Round-1 dev parent | `0bb00beaf` |
| dev commits merged | **75** |
| Working tree | clean except `progress.md`; verified absent from the commit |

Drift, measured rather than carried: **2** mover bodies edited (not 11),
**2** new dev screen methods, and — new this round — **1** dev method
*deleted* (`_library_media_int_backing_id`, promoted to a module-level
function).

---

## R2.1 Conflicts — 7 hunks / 3 files

The coordinator counted 4 screen hunks; `--conflict=diff3` splits the
`_library_media_layout_signature` region into two, so the record is 5.

| # | Enclosing | base/ours/theirs | Resolution |
|---|---|---|---|
| 1 | `_library_media_layout_signature` | 2/2/12 | **COMBINED** — dev's new canonical-width body, three receivers swapped |
| 2 | `_library_media_layout_signature` | 2/2/2 | **COMBINED** — dev's `canonical.reader_width` + our `_media_state.reader_preferences` |
| 3 | `_select_library_media_reader_row` | 57/2/62 | OURS (delegator); dev's +5 ported |
| 4 | `_review_cursor_for_display` | 2/2/2 | **COMBINED** — dev's `library_media_int_backing_id(...)` wrapper + our receiver |
| 5 | `handle_library_media_review_selected` | 25/1/27 | OURS (delegator); dev's +2 ported |

`test_library_media_render_fixes.py` (1 hunk): dev replaced a literal
assertion string with `_NO_PROVIDER_REASON`; we had retargeted the line
below. Kept both, dev's assertion value byte-for-byte.

`test_library_media_return_settlement.py` (1 hunk): dev switched to its new
`canonical_reader_width`; we had retargeted both receivers. Kept both.

**Three of the seven hunks were genuinely two-sided** — up from one in round
1. Taking either side whole would have silently lost the other's work, and
the ratio is worth noting: as dev keeps editing the cluster this branch
moved, the share of conflicts that cannot be resolved by picking a side goes
up, not down.

### Inverse-transform verification

Rather than eyeballing the combined hunks, each was checked by applying the
*declared* substitution to dev's own version and comparing:

```
IDENTICAL  _library_media_layout_signature             (dev + declared subs == ours)
IDENTICAL  _restore_library_media_reader_width_on_open (dev + declared subs == ours)
IDENTICAL  _review_cursor_for_display                  (dev + declared subs == ours)
```

And for the test files, against dev's file plus the whole-family receiver
swap:

```
IDENTICAL  Tests/UI/test_library_media_return_settlement.py
IDENTICAL  Tests/UI/test_library_media_side_by_side.py
DIFFERS    Tests/UI/test_library_media_render_fixes.py     (2 lines)
DIFFERS    Tests/UI/test_library_media_reader_shell.py     (1 line)
```

All three residual differences were read: two are PROSE retargets from
wave-7 task 3's stale-doc sweep, one is a `next_screen.` receiver the
`screen.`-anchored swap does not cover. None is a round-2 change. This is the
round-1 reviewer's own technique (undo the declared transform, compare to the
parent) applied by the implementer rather than waiting for review.

---

## R2.2 The two ports

Same precondition, checked first: both controller bodies were still
byte-identical to dev's pre-edit version, so the ports are wholesale
replacements — exact by construction.

```
=== ROUND-2 PORT IDENTITY ===
  text=IDENTICAL ast=IDENTICAL  _select_library_media_reader_row
  text=IDENTICAL ast=IDENTICAL  handle_library_media_review_selected
```

Running total across both reconciliations: **13 ports, 13 byte-identical, 0
substitutions.**

### Two names, two different kinds — and only one is a binding

`_restore_library_media_reader_width_on_open` is dev's new screen method,
bound as the **ninth** group-(e) late-binding callable. Arity 91 → 92, and
the construction site verified equal to the signature with no missing or
extra name.

`library_media_int_backing_id` is **not a binding at all.** Dev promoted it
out of a `LibraryScreen` method into a module-level function in
`Library/library_media_state.py`, so the ported body reads it as a **bare
module global**, which resolves against the *controller module's*
`__globals__` — recipe §3's module-globals-coupling shape. No `self.<attr>`
census can see it. Unbound, it is a `NameError` on every "Review selected"
press. Imported into the controller with a comment recording why.

An AST resolver over the whole controller — class members, `__init__`
assignments, the 82 generated shims, *and* module globals — reports zero
unbound names of either kind after both fixes.

---

## R2.3 Census over the merged tree

All five spellings, 82 names re-derived from `LibraryMediaState`.

**Spellings 1–4:** 439 hits at first pass. **Sixteen live references** dev's
75 commits added were retargeted — 6 in the screen, 11 in tests (one file
overlapping both counts is counted once per site).

The six screen-side ones all sit in dev's new task-31979 width-restore work:

| Line | Shape |
|---|---|
| 7480 | read — `reader_preferences` |
| 7481 | read — `reader_layout` |
| 7482 | read — `view` |
| 7484 | read — `reader_layout` |
| **7487** | **WRITE — `reader_layout = layout`** |
| 7543 | read — `view` |

**Line 7543 is the one worth the entry.** It sits inside
`_sync_library_media_reader_layout_from_shell`, a body this wave had
*already* retargeted — the surrounding lines read `self._media_state.view`,
`self._media_state.reader_preferences`, `self._media_state.reader_layout`.
Dev added one new `reader_has_item=self._library_media_view == ...` kwarg in
the middle of them, and git merged it in **without a conflict**. A reviewer
reading the conflict hunks would never see it. That is the whole case for
running the census over the merged TREE rather than over the diff.

**Test-side:** 11 receiver retargets across three files, plus two
`test_review_set_walker.py` fakes driving `handle_library_media_review_
selected` — a MOVED name. This is the second instance of the shape round 1
met with `handle_library_media_reader_more`, and it was resolved the same
way: a minimal `_ReviewSelectedControllerFake` reproducing the one production
hop (two generated shims, one group-(e) property, one framework service), so
the real controller body still runs. Both fakes are green.

**Spelling 5:** 3 candidates, identical to round 1 — the `canvas_sync.py`
site (dotted media branch read live, survived dev's edits), the shim
mapping's own definition, and the CCP false positive.

**Final state:** 420 hits; the only code-classified references outside the
controller are the two test fakes' deliberate shim properties.

---

## R2.4 Pruned delegators — zero new callers

Censused in **both** spellings, per round 1's own gap note:

```
22 PRUNED NAMES, ALL SPELLINGS
  library_media_controller.py                  63   (owner)
  test_library_media_wiring.py                 44   (the pin)
  test_library_media_viewer_speaker_rename.py   7   round 1's retarget, via _media_controller
  test_local_media_reading_service.py           2   prose
  library_media_state.py                        1   prose

BOUND screen.<mover>(...) calls in Tests/: 95 — and NOT ONE is a pruned name
```

Structural cross-check, independent of the ratchet: **all 118 non-pruned
mover delegators present on `LibraryScreen`, every one forwarding to
`_media_controller`, and no pruned name reappeared.**

---

## R2.5 Pins

| Row | Before | After |
|---|---|---|
| `library_screen.py` — `LibraryScreen` | `(35626, 1289)` | **`(35743, 1290)`** |
| `library_media_controller.py` | `4612` | **`4630`** |
| `library_media_browse_controller.py` | `371` | **unchanged** (dev now 686) |
| `library_unavailable_navigation.py` | `817` | unchanged |

The method SET check is exact in **both** directions, and this round it
finally exercises the second direction for real: `merged − branch` is dev's
two new names, `branch − merged` is the one name **dev itself deleted**. Net
+1. A count-only check would have read `+2 −1 = +1` identically for a lost
delegator paired with a gained one.

Line delta **+117** against dev's own **+121**: `121 − 7` (the two ported
mover edits, which the screen does not take) `+ 3` (the construction-site
kwarg) `= 117`. Controller **+18** = 7 ported + 6 binding + 5
import-and-comment. Both exact.

`library_media_browse_controller` refused for the **fifth** time: 410 at wave
6, 649 at round 1, **686** twelve hours later at round 2, against a pin of
371. Dev is still adding to it.

---

## R2.6 Battery

| Suite | Result |
|---|---|
| 8 wiring suites + media characterization | **62 passed, 0 failed** |
| both size ratchets | 42 passed, 4 failed (same four dev-owned rows as round 1) |
| ADR-055 interlock + canvas-sync dotted-path guard | **2 passed, 0 failed** |
| `./scripts/preflight.sh` | **all seven checks passed** |
| dev's 9 changed media test files | 652 passed, 21 failed |

Baseline: an isolated `origin/dev` worktree at `37bf45fb6` with its own venv,
parity **proven before any count was read** — Python 3.14.2 both sides,
`uv pip list` identical at 106 packages, each tree verified to resolve its
own `tldw_chatbook` from its own root.

```
branch    21 failed / 652 passed
baseline  21 failed / 652 passed
-> 20 SHARED, 1 branch-unique, 1 baseline-unique
```

**The equal counts prove nothing** — recipe §7's own environment note — so
the comparison is on failing NAME sets. The single branch-unique name,
`test_media_trash_back_and_escape_restore_distinct_media_return[escape]`, is
already on TASK-31249 as load-sensitive, so it got §7's **third disposition
level** rather than a verdict: an INTERLEAVED round-robin across the two
trees, alternating run by run rather than a sequential batch.

| | fail rate, interleaved |
|---|---|
| merged branch | **1 / 5** |
| `origin/dev` baseline | **2 / 5** |

Nondeterministic on both trees and worse on dev's. **Zero real branch-unique
failures.** The baseline-unique name
(`test_trash_back_exact_scroll_precedes_captured_control_focus`) is the same
one round 1's pairing surfaced, in the same return-settlement family.

---

## R2.7 The backlog id collision

`preflight` caught it: **dev minted TASK-31976 inside these 75 commits**,
colliding with round 1's filing of the media exclusion debt.

This is not a near-miss of the hygiene rule — it is the rule's own documented
failure mode reproducing exactly. Both sessions swept for a maximum before
filing; both observed the same pre-merge maximum from different worktrees, so
both leapfrogged into the same block. `lessons-backlog-hygiene.md` records
this from 2026-08-21 (TASK-19573/TASK-19601) and says plainly why leapfrogging
further does not help: the race is not "did you leapfrog far enough", it is
"did anyone else observe the same maximum before you merged."

Applying the tie-break rule that replaced leapfrogging — **the older
`created_date` keeps the id regardless of status**:

| | created | outcome |
|---|---|---|
| dev's "Recover Console sends after trace boundary construction failure" | 2026-09-07 **20:53** | **keeps TASK-31976** |
| ours, the media exclusion debt | 2026-09-07 **22:06** | **renumbered to TASK-32013** |

New id taken after a fresh sweep across every remote ref and every local
backlog directory (observed maximum 32012). A `## Renumbering provenance`
section was added, and both inbound references moved with it (TASK-31249's
cross-pointer, and `test_review_set_walker.py`'s fixture comment).

**Two references were deliberately NOT rewritten:** the round-1 merge
(`3c11f902a`) and follow-up (`2c4c4d709`) commit messages name TASK-31976 and
are published history on PR #2501. Rewriting them would falsify a record
rather than fix a pointer; the provenance section is the bridge. Preflight is
green at 3421 task files.

---

## R2.8 A third census gap

`Tests/Architecture/test_library_media_wiring.py`'s
`_MEDIA_CONTROLLER_BOUND_NAMES` coverage check is **one-directional**. It
asserts the 92 listed names each resolve to a `property` on the controller,
and that the list has no duplicates. It never asserts the converse — that
every name a moved body references is *in* the list.

Measured consequence: this branch has added **nine** group-(e) bindings
across two reconciliations, and that check passed unchanged before and after
every one of them, because the new names simply were not in its tuple. What
actually caught both rounds' missing bindings was an ad-hoc AST resolver run
by hand. And the module-global shape (§R2.2) it cannot see at all.

Recorded on TASK-32013 with the fix shape, **not fixed here** — the same
reasoning that declined to revert a mover or restore a delegator inside a
merge commit.

---

## R2.9 Concerns

1. **The mitigations do not transfer between rounds, and the rate is
   rising.** Round 1 absorbed two dev-side instances in 306 commits; round 2
   absorbed two more in 75 — plus a module-globals coupling and an id
   collision that round 1 did not see at all. Three of seven conflicts were
   two-sided here versus one of eighteen in round 1. The next reconciliation
   should budget for the class, not for the count.
2. **`library_media_browse_controller` is at 686 against a 371 pin**, refused
   for the fifth time. It gained 37 lines in the twelve hours between the two
   reconciliations. The refusal is still right and it is still nobody's.
3. **The one-directional wiring check (§R2.8) is the most likely place the
   next merge ships a real defect.** Both reconciliations found real missing
   bindings that the full battery reported green on.
4. **`test_media_trash_back_and_escape_restore_distinct_media_return[escape]`
   and `test_trash_back_exact_scroll_precedes_captured_control_focus`** are
   both nondeterministic on both trees, in the same return-settlement family,
   and have now each surfaced as "unique" in one of the two rounds' pairings.
   Anyone reading a single run of either will be misled.
