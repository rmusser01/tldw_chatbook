# Task 1 report — Controller-file size governance (task-31203 AC#4)

Plan: `Docs/superpowers/plans/2026-09-03-library-decomposition-wave3-search-rag.md`, Task 1.
Driving AC: `backlog/tasks/task-31203 - Library-decomposition-wave-3-combined-searchRAG-series.md`, AC#4.
Model studied: `Tests/Architecture/test_screen_size_ratchet.py`.

## Decision + reasoning

**Chosen: option (a)** — exact per-file `_BUDGETS` rows in a new sibling guard,
`Tests/Architecture/test_library_modules_size_ratchet.py`, **with one addition beyond
the screen ratchet's own design**: the controller set is discovered by glob
(`UI/Library_Modules/*_controller.py`) at collection time rather than hand-listed, so
an unlisted file fails loudly instead of silently landing ungoverned — the
self-defending property `test_screen_size_ratchet.py`'s hand-kept dict does not have
(and whose absence is exactly how `library_screen.py` itself went ungoverned for a
month while tripling in size, per that file's own docstring).

Options (b) aggregate budget and (c) loose per-file tolerance were rejected:

- **(b) aggregate**: cannot localize which controller grew, so review sees only "the
  total moved" with no signal distinguishing subsystem X's sanctioned move from
  subsystem Y's creep — the exact ambiguity per-file governance removes. It would also
  concentrate merge-conflict surface from every concurrent controller PR onto one
  shared number, multiplying the "dev races" contention the wave-3 plan already flags
  for the screen's single row.
- **(c) loose tolerance**: no single tolerance value serves both purposes across
  controllers ranging ~280–2,000+ lines — loose enough to survive a large sanctioned
  move does nothing to catch creep between moves (the actual failure mode), and tight
  enough to catch creep fails on every sanctioned move, reproducing the "ratchet only
  ever goes down" problem the plan's own design-tension note calls out.

**Resolving the byte-for-byte-canon tension**: re-reading how the screen ratchet is
actually *operated* (not just its docstring's "may only ever go DOWN" framing) shows
its `_BUDGETS` row for `library_screen.py` was RAISED twice during the wave-2
final-review fix wave, each time with a dated justification comment. The screen
ratchet's real enforcement is two checks, not one hard one-way rule — a **ceiling**
(`test_*_does_not_grow_past_its_budget`) plus an **anti-slack bound**
(`test_budget_is_not_left_slack_after_a_*`). Neither test can read intent, but
together they make both "sanctioned move" (ceiling raised, in the same diff, next to
the method bodies that justify it) and "creep" (ceiling breached with no
corresponding row edit) visible in code review. Task 1 reused this existing,
already-proven-in-practice model rather than inventing a stricter one — same two-check
shape, applied to a new file set, with re-pinning happening in the same commit as
every sanctioned move (identical to recipe §6).

**Method count: deliberately NOT tracked** (unlike the screen ratchet, which tracks
line count + method count of one dominant class). The screen ratchet's method count
exists to catch a class made shorter by *compressing* bodies rather than by extracting
responsibility — meaningful only because `ChatScreen`/`LibraryScreen` are each exactly
one class filling their whole file, always named after the screen. Controller files
under `Library_Modules/` are not shaped that way: the byte-for-byte canon's
constructor-dependency-binding pattern deliberately produces small immutable helper
classes in the SAME file as the primary controller — Protocol ports, request
"fences," result "receipts," outcome snapshots (e.g. `CaptureRequestFence`/
`CaptureArchiveReceipt` in `library_collections_capture_controller.py`; the `*Port`
protocols in `library_notes_sync_controller.py`). There is also no reliable
filename→class-name convention: `library_skill_import_controller.py`'s primary class
is `LibrarySkillImportCoordinator`, not `...Controller` — a naive filename-derived
lookup would silently miss it. Picking "the" dominant class per file would need either
a hand-maintained override table (reintroducing the non-self-defending problem this
design otherwise avoids) or summing methods across every class in the file (counting
the helper-class proliferation the canon itself encourages as controller-growth,
punishing a pattern the recipe recommends). File line count has neither problem and is
the exact axis the wave's own design tension is stated in, so it is the only metric
tracked. Recorded as a row-by-row override option for the future, not needed by any of
the twelve current rows.

**Scope**: `*_controller.py` only, per AC#4's own framing (it names controller files
by example). `Library_Modules/`'s state files and other support modules are out of
scope for this task.

Full decision + reasoning also recorded in `backlog/docs/library-decomposition-recipe.md`,
new §17 ("Controller-file size governance (task-31203 AC#4)").

## Measured rows (pinned at exact values, 2026-09-03)

All twelve `*_controller.py` files under `UI/Library_Modules/` at landing:

| Controller file | Lines |
|---|---:|
| `library_collections_capture_controller.py` | 699 |
| `library_collections_controller.py` | 1,689 |
| `library_conversation_reader_controller.py` | 943 |
| `library_conversations_controller.py` | 1,738 |
| `library_export_controller.py` | 1,307 |
| `library_media_browse_controller.py` | 371 |
| `library_media_trash_browse_controller.py` | 319 |
| `library_note_import_controller.py` | 587 |
| `library_notes_sync_controller.py` | 2,023 |
| `library_prompt_browse_controller.py` | 281 |
| `library_skill_import_controller.py` | 760 |
| `library_skills_browse_controller.py` | 413 |
| **Total** | **11,130** |

Measured with the guard's own `_measure` expression:
`len(path.read_text(encoding="utf-8").splitlines())` — matches `wc -l` here exactly
(no missing-trailing-newline discrepancy on any of the 12 files).

## TDD sequence

1. Wrote the guard's docstring + `_measure`/`_discovered_controller_paths` machinery
   and the three test functions first, with the real `_BUDGETS` dict already filled
   in at exact measured values (there was no meaningful "RED" state to author here
   before any rows existed — the glob-discovery test itself IS the mechanism that
   would have failed against an empty dict, verified directly by mutation below
   rather than via an initial empty-dict authoring step).
2. Ran the full file: **25 passed** (1 discovery test + 12 ceiling + 12 anti-slack) —
   confirms exact-pass at zero slack for every row.
3. Ran the four mutations below to prove each failure mode fires correctly in both
   directions, then reverted each one and re-confirmed the clean 25-passed state.
   `git diff` on the test file after the full mutation round-trip is empty (confirmed).

## Mutation evidence (both directions + the self-defending property)

**1. Unlisted EXISTING file** (deleted the `library_prompt_browse_controller.py` row
from `_BUDGETS`, file still on disk):

```
AssertionError: New Library_Modules controller file(s) with no _BUDGETS row:
['tldw_chatbook/UI/Library_Modules/library_prompt_browse_controller.py'].
...
1 failed, 22 passed
```

(22, not 24 — that row's own ceiling/slack parametrizations disappeared along with the
dict entry, since both parametrize over `sorted(_BUDGETS)`.) Reverted; back to 25
passed.

**2. Genuinely NEW file** (dropped a throwaway
`_mutation_test_scratch_controller.py` into `Library_Modules/`, matching the glob,
not touching `_BUDGETS` at all):

```
AssertionError: New Library_Modules controller file(s) with no _BUDGETS row:
['tldw_chatbook/UI/Library_Modules/_mutation_test_scratch_controller.py'].
...
1 failed, 24 passed
```

This is the property `test_screen_size_ratchet.py`'s hand-kept dict does not have —
confirmed directly, not just argued. Deleted the scratch file; back to 25 passed.

**3. Growth trip** (lowered `library_skills_browse_controller.py`'s budget from 413 to
400, 13 below its real measurement):

```
AssertionError: tldw_chatbook/UI/Library_Modules/library_skills_browse_controller.py
grew to 413 lines (budget 400, +13).
...
1 failed, 24 passed
```

Reverted to 413; back to 25 passed.

**4. Anti-slack trip** (raised `library_prompt_browse_controller.py`'s budget from 281
to 332, i.e. 51 lines over — one past the 50-line tolerance). Re-run and captured fresh
(round-4 evidence requested in review) rather than relied on from memory:

Row change: `"...library_prompt_browse_controller.py": 281,` → `281 + 51,`

```
$ .venv/bin/python -m pytest \
    "Tests/Architecture/test_library_modules_size_ratchet.py::test_budget_is_not_left_slack_after_a_move" -q

FAILED Tests/Architecture/test_library_modules_size_ratchet.py::test_budget_is_not_left_slack_after_a_move[tldw_chatbook/UI/Library_Modules/library_prompt_browse_controller.py]
E       AssertionError: tldw_chatbook/UI/Library_Modules/library_prompt_browse_controller.py is 51 lines under its budget (281 vs 332). Set it to 281 so the real measurement is what's pinned.
E       assert (332 - 281) <= 50
1 failed, 11 passed
```

(11, not 12 — only this row's own anti-slack parametrization is in scope when the
test id is filtered to a single test function.)

**Boundary check, same round** (row set to `281 + 50` instead — exactly the tolerance):

```
$ .venv/bin/python -m pytest \
    "Tests/Architecture/test_library_modules_size_ratchet.py::test_budget_is_not_left_slack_after_a_move" -q

12 passed, 94 warnings in 0.43s
```

Confirms `<=` is the intended inclusive boundary: 50 over passes, 51 over fails.
Reverted to `281`; `git diff --stat Tests/Architecture/test_library_modules_size_ratchet.py`
empty, and the full file re-run at `25 passed` again before proceeding.

## Battery

- **New guard alone**: 25 passed (0 failed).
- **New guard + screen ratchet together**
  (`Tests/Architecture/test_library_modules_size_ratchet.py
  Tests/Architecture/test_screen_size_ratchet.py`): 28 passed, 2 failed — both
  failures are `chat_screen.py`'s pre-existing, already-documented red
  (`test_screen_does_not_grow_past_its_budget[...chat_screen.py]` and
  `test_task_22507_4_does_not_worsen_chat_screen_base`), unrelated to this task.
  Confirmed pre-existing by running the unmodified screen-ratchet file alone before
  any change in this task landed: identical 2 failed / 3 passed, chat_screen measured
  at 20,538 lines against a 16,966-line budget (+3,572) — this task neither caused nor
  touched that gap; it is concurrent Console-side growth the screen ratchet's own
  docstring already documents as a known, expected class of drift.
- **Library recompose census** (`Tests/UI/test_library_recompose_ratchet.py`): 6
  passed, 0 failed.
- **`./scripts/preflight.sh`**: all five derived-artifact checks passed clean (CSS
  bundle sync, profile-owned-path census, production diagnostic inventory, duplicate
  backlog task ids, chachanotes table allowlist / index plan pins) — no drift from
  this task's changes.
- **`Tests/Architecture` full suite** (`-p no:randomly`): **527 passed, 15 failed, 1
  skipped** (328.71s). `test_library_modules_size_ratchet.py`'s own 25 tests are all
  in the 527 passed. The 15 failures span `test_console_review_selection_controller_
  boundary.py` (1), `test_console_wave6_closeout_inventory.py` (1),
  `test_console_wave6_inventory.py` (2), `test_default_timeout_session_guard.py` (1),
  `test_persistent_diagnostic_inventory.py` (2), `test_screen_size_ratchet.py` (2 —
  the already-documented chat_screen reds), `test_timer_path_static_update_
  inventory.py` (3), `test_worker_exclusive_group_inventory.py` (2) — none touch
  Library_Modules, this guard, or anything this task changed. Confirmed pre-existing,
  not introduced by this task: `git stash -u` (removing every change this task made)
  and re-running the 12 non-chat_screen failures reproduced the identical 12 failures
  against the unmodified base (39.19s); the 2 chat_screen failures were independently
  confirmed pre-existing earlier by running `test_screen_size_ratchet.py` alone before
  this task's first edit (2 failed/3 passed, chat_screen measured 20,538 lines vs a
  16,966-line budget). Stash was popped immediately after, restoring this task's
  changes.

## Files changed

- `Tests/Architecture/test_library_modules_size_ratchet.py` — new. The guard itself:
  module docstring (design tension, decision, why line-count-only, self-defending
  glob), `_BUDGETS` (12 rows), `_measure`, `_discovered_controller_paths`, and three
  tests (`test_every_controller_file_has_a_budget_row`,
  `test_controller_does_not_grow_past_its_budget`,
  `test_budget_is_not_left_slack_after_a_move`).
- `backlog/docs/library-decomposition-recipe.md` — new §17 (Controller-file size
  governance): decision + reasoning, re-pin-at-move flow, why-line-count-only
  rationale, measured-rows table, scope note, mutation-evidence summary.
- `backlog/tasks/task-31203 - Library-decomposition-wave-3-combined-searchRAG-series.md`
  — AC#4 checked; Implementation Notes added, scoped explicitly to AC#4 only. Task
  status left at "To Do" (AC#1–3, the search+RAG extraction series itself, are out of
  this task's scope and remain undone).

## Self-review

- Followed the steer's directive (option (a), glob-based self-defending discovery)
  after independently re-deriving the same conclusion from the plan's three listed
  options and the model file's own documented practice (raises-with-justification,
  not a hard one-way rule) — the steer and the independent read agreed, so no
  deviation to flag.
- The "method count" question was explicitly weighed and answered with a concrete,
  file-level counterexample (`LibrarySkillImportCoordinator`) rather than asserted —
  checked by grepping every controller file's class defs before deciding, not assumed
  from the screen model's shape.
- Mutated in all four directions the task asked for (growth, slack, unlisted-existing,
  unlisted-new) plus verified the tolerance boundary itself (50 passes, 51 fails), and
  confirmed a clean revert via `git diff` before treating any measurement as final —
  no mutation was left in the committed state.
- Did not touch `library_screen.py`, `chat_screen.py`, or any controller's production
  code — this task is guard-and-docs only, matching its AC's scope.
- One thing NOT done that could be argued for: no `progress.md` "Task 1: complete"
  ledger line was added proactively before this report — added alongside this report
  per the SDD ledger's own established one-line-per-task convention (see the wave-2
  ledger's `Task N:` entries), not part of the explicit dispatch instructions but
  cheap and consistent with house style.

## Fix round 1 — restored task-31203's stripped Renumbering provenance section

**Finding (Important, from review):** the `backlog task edit 31203 --check-ac 4
--notes "..."` call used to tick AC#4 and add Implementation Notes silently deleted
the task file's `## Renumbering provenance` section (the TASK-27021→31203 collision
record documented at the top of the task file). This is exactly the trap
`backlog/docs/lessons-backlog-hygiene.md` already documents — line 175 ("the CLI
strips some free-form sections") and lines 179–188 (the `--notes` command replaces
the whole `NOTES` block and the file's other free-form content, not just appends;
prescribed mitigation: "diff the task file after any `--notes` command"). I ran the
`--check-ac`/`--notes` call and moved on without diffing the file against its prior
state afterward — the exact step the lesson calls out, skipped.

**Fix applied:**

1. Restored `## Renumbering provenance` verbatim from the base commit
   (`git show 2a90fa74c:"backlog/tasks/task-31203 - ...md"`), placed back in its
   original position (after Acceptance Criteria), with the new `## Implementation
   Notes` section appended after it — preserving both the historical record and this
   task's own AC#4 notes.
2. Diffed the restored file against the same base commit:
   ```
   diff <(git show 2a90fa74c:"backlog/tasks/task-31203 - ...md") \
        "backlog/tasks/task-31203 - ...md"
   ```
   Confirmed only three differences remain: `updated_date` (07:30 → 20:49), the AC#4
   checkbox (`[ ]` → `[x]`), and the appended `## Implementation Notes` block. Nothing
   else — including the Renumbering provenance section itself — differs from base.
3. Round-4 mutation (anti-slack trip) re-run and captured fresh per the review's Minor
   — see the updated §Mutation evidence item 4 above (both the 51-over failure and the
   50-over boundary pass now have literal captured command output, not a "verified
   interactively" claim).

**No new lesson added** — `lessons-backlog-hygiene.md` already documents this exact
trap (the free-form-section-stripping note at line 175 and the `--notes`-replaces
note with its diff-after-notes mitigation at lines 179–188); this incident is a
second confirmation of a lesson already on the books, not a new one. Adopted for the
rest of this task and future ones: **diff the task file against its pre-edit git blob
immediately after any `backlog task edit --notes` (or `-s Done --notes`) call**,
before treating the edit as final — added to my own working checklist, not just
noted here.

Commit: `fix(backlog): restore task-31203 renumbering provenance stripped by the notes edit`.
