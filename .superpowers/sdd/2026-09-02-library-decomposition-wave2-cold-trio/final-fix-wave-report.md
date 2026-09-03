# Wave-2 final review — fix wave report

Base: `09a5cadff` (wave-2 close). Branch: `refactor/library-decomp-wave2-cold-trio`.
Scope: the final review's complete findings list (0 Critical, 2 Important, 4
Minor) — nothing else touched.

## Findings addressed

### 1. Durable record (Important #1)

Two parts, both done:

**(a) Track the SDD directory.** `git add -f
.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio/` — 22
files (progress ledger, 10 review diffs, 10 task reports, 1 task brief pair
already counted). Verified tracking works the same way the repo's existing
77 tracked `.superpowers/sdd/` files do: `git ls-files .superpowers/sdd |
wc -l` went 77 → 99 after staging. `.superpowers/` itself stays
`.gitignore`d (line 8); `git add -f` overrides per-path, matching the
existing precedent directories (`2026-07-30-watchlists-briefings-phase-1`,
`2026-08-22-console-library-controls`, etc.).

**(b) Copy the guard-gap note into the recipe.** Added as item 5 of `backlog/
docs/library-decomposition-recipe.md` §16's "Lessons" list (the closest
existing home to a dedicated "guards" section — §16 is the wave-2-close
summary and its Lessons list is where prior guard-shaped findings, e.g. the
RED-commit criterion ruling, already live). The note states plainly: the
wiring cluster constants (`_EXPORT_CLUSTER_METHOD_NAMES`, 22 entries;
`_COLLECTIONS_CLUSTER_METHOD_NAMES`, 64 entries — both counts verified by
AST-walking the two wiring test files, not eyeballed) are hand-written
tuples frozen at PR time from a one-time census, and nothing re-runs that
census at test time to catch a NEW same-named method landing on
`LibraryScreen` later. A future 23rd export-named or 65th collections-named
method slips every existing guard. Distinguished explicitly from the
DIFFERENT guard that does exist (the recompose census's anti-slack check,
task-27019) so a reader doesn't conflate the two axes.

**Citation check**: recipe §8 (line ~462) and §16 (line ~1557) already cite
`.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio/
task-8-report.md` — verified this is byte-identical to the path now
tracked by (a), so no citation edit was needed. `backlog/tasks/task-31203
...md`'s own description references "the wave-2 SDD task-8 report" without
a literal path, so nothing to fix there either.

### 2. Stale mechanics comments (Minor #3)

- `tldw_chatbook/UI/Library_Modules/library_collections_state.py:132-135` —
  the comment claimed the three entangled fields (`reader_preferences`,
  `reader_persistence_locks`, `reader_layout`) are still "routed into this
  dataclass's field by the screen's property shim." That shim was deleted
  wholesale in the collections cleanup PR (wave-2 task 7). Verified the
  actual post-cleanup mechanism by reading `library_screen.py`'s
  `__init__`: all three are direct attribute writes onto
  `self._collections_state.<field>` (the `reader_preferences` tuple-unpack
  at ~:2368, `reader_layout` at ~:2366→2378ish, `reader_persistence_locks`
  at ~:2434) — no shim anywhere in the path. Comment rewritten to describe
  that directly.
- `tldw_chatbook/UI/Library_Modules/library_export_state.py:~42` (module
  docstring, `origin_row_id` paragraph) and `~146-149` (the field's own
  inline comment) — both still said the field's value is supplied "through
  the generated property shim." The export cleanup PR (wave-2 task 4)
  deleted that shim too; verified `library_screen.py` now reads/writes
  `self._export_state.origin_row_id` directly (read at ~:3472, cleared at
  ~:23267). Both comments rewritten to state the direct-attribute reality
  and name which task removed the shim.

`library_export_state.py`'s OTHER "placeholder default" comment (the
`form` field, ~85-88) was left untouched — it already accurately says
`form` is passed as a constructor argument, which matches
`library_screen.py:3288`'s `LibraryExportState(form=self._default_
library_export_form())` call. Not in scope per the review's line
citations, and not stale.

Neither file is `_BUDGETS`-pinned (only `chat_screen.py` and
`library_screen.py` are), so these edits needed no ratchet accounting.

### 3. Construction-order sentinel (Minor #4)

Added an 9-line comment immediately above `self._export_controller =
LibraryExportController(` at `library_screen.py:2276` (now :2276, comment
precedes it) stating: `self._export_state` does not exist yet at this
point in `__init__`; it is constructed later at ~:3288 specifically to
preserve the computed `form` default's original `__init__` evaluation
position; every dependency passed to the controller is a lazy accessor
(`lambda`, not a bound value); no controller method may run during
`__init__`; an eager `export_state_accessor()` call from here would raise
`AttributeError`. Comment-only, verified no code changed (`git diff`
shows 9 pure insertions on that file, 0 deletions).

### 4. Wave-3 size-governance note (Minor #6)

Added AC #4 to `backlog/tasks/task-31203 - Library-decomposition-wave-3-
combined-searchRAG-series.md` via `backlog task edit 31203 --ac ...`:
records that `Library_Modules` controller files have no size-ratchet
governance today (`library_collections_controller.py` measured 1,689
lines, `library_conversations_controller.py` 1,738 lines — both verified
via `wc -l` against the live tree, not carried from a stale report), and
that wave 3 must record an explicit decision (add `_BUDGETS`-style rows,
an equivalent mechanism, or a reasoned defer) rather than leave the
question unaddressed.

## Pin accounting (library_screen.py)

Only finding 3 touched `library_screen.py`; findings 1, 2, and 4 touched
no `_BUDGETS`-pinned file. The construction-order sentinel comment added 9
lines (`git diff --stat`: `9 insertions(+)`, confirmed — not the 8 my own
line-by-line count first suggested).

Fresh `_measure()`-equivalent re-run (same `len(source.splitlines())` +
AST method count `test_screen_size_ratchet.py` itself uses):

```
lines: 42420
methods: 1267
```

`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]` raised in this
same commit from `("LibraryScreen", 42411, 1267)` to `("LibraryScreen",
42420, 1267)` — methods unchanged (comment-only edit, no new
`FunctionDef`), with a dated justification comment following the
foundation run's own task-8 precedent (a strengthened comment there once
pushed the file 24 lines over its just-set ceiling; the resolution was to
re-measure and raise with a dated comment, not to strip the comment or
leave the ceiling red). Net wave-2 trajectory is still down: 43965
(wave-2 start, `2b20ebbb9`) → 42420 now, a 1545-line shrink despite this
fix wave's own 9-line increase.

## Verification

- `.venv/bin/python -m pytest Tests/Architecture/test_library_export_wiring.py
  Tests/Architecture/test_library_collections_wiring.py
  Tests/Architecture/test_library_conversations_wiring.py
  Tests/Architecture/test_screen_size_ratchet.py -p no:randomly -q`:
  **19 passed, 2 failed** — the 2 failures are
  `test_screen_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/
  chat_screen.py]` and `test_task_22507_4_does_not_worsen_chat_screen_base`,
  both pre-existing/documented (unrelated `chat_screen.py` drift on `dev`,
  not touched by this fix wave). Both `library_screen.py` ratchet rows
  (ceiling and no-slack) pass at the raised pin.
- `./scripts/preflight.sh`: all six checks green (CSS bundle, profile-owned
  path census, production diagnostic inventory, backlog task ids — "No
  duplicate task IDs across 2924 task files" — chachanotes table allowlist,
  index plan pins).

## Files changed

- `.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio/*`
  (22 files, force-added; this report is the 23rd)
- `backlog/docs/library-decomposition-recipe.md` — guard-gap note (§16
  Lessons, item 5)
- `backlog/tasks/task-31203 - Library-decomposition-wave-3-combined-
  searchRAG-series.md` — AC #4 (size-governance note)
- `tldw_chatbook/UI/Library_Modules/library_collections_state.py` — stale
  comment rewrite (~132-135)
- `tldw_chatbook/UI/Library_Modules/library_export_state.py` — stale
  comment rewrites (~42, ~146-149)
- `tldw_chatbook/UI/Screens/library_screen.py` — construction-order
  sentinel comment (~2276), +9 lines, 0 code changes
- `Tests/Architecture/test_screen_size_ratchet.py` — `library_screen.py`
  budget row raised 42411→42420 (lines only; methods unchanged), dated
  justification comment added

## Concerns

None blocking. The guard gap documented in finding 1(b)/recipe lesson 5
remains genuinely open (deferred, not closed) — closing it needs an active
AST re-census guard, which is explicitly out of this fix wave's scope and
is now flagged in the recipe for wave 3 or a dedicated follow-up.
