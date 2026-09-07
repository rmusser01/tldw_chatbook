---
id: TASK-16196
title: Delete the dead legacy Study event-handler module
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 03:05'
updated_date: '2026-08-15 22:36'
labels:
  - cleanup
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Event_Handlers/Study_Events/study_events.py`'s `STUDY_BUTTON_HANDLERS` / `study_event_handler` are referenced nowhere outside the module at current dev HEAD — flashcard handling moved to `Study_Modules/flashcards_handler.py` during the Study rebuild, and TASK-15471's implementer and reviewer independently verified the unreachability (grep + dispatch-path read). The module still contains synchronous ChaChaNotes writes that the input-latency audit flagged — dead code shaped like a loaded gun. Delete the dead handler surface (owner ruling: delete dead code rather than leave loaded guns), preserving anything still genuinely imported (verify each symbol, not just the table). Coordinate with TASK-16195, which may want one piece of it (add-topic) resurrected in the modern location first. Surfaced during TASK-15471 (per-click I/O off-loop, PR #1625 merged `172ada448`) and its concurrency review; evidence in the session review record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unreferenced handler table and its dead handlers are removed
- [x] #2 Anything still imported elsewhere is identified and preserved (evidence: import graph or grep per symbol)
- [x] #3 Test collection and the Study suites stay green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fast-forward the worktree onto `origin/dev` so 16195's merge (PR #1681, `19b0c1a03`) is on the base, since its removal of `#add-topic-btn` from `Study_Window.py` changes what the legacy module's `handle_add_topic` maps onto.
2. Per-symbol import-graph sweep of `Event_Handlers/Study_Events/study_events.py`: for each of the 3 event classes, `StudyEventHandler`, `study_event_handler`, `STUDY_BUTTON_HANDLERS`, and each of the 8 handler methods, grep `tldw_chatbook/` + `Tests/` for every reference outside the module itself, and check whether the package `__init__.py`'s re-exports (`StudyCardCreatedEvent`, `StudyCardReviewedEvent`, `StudyTopicSelectedEvent`, `StudyEventHandler`) have any external importer of the `Study_Events` package.
3. Cross-check the remaining `STUDY_BUTTON_HANDLERS` button ids (`add-child-btn`, `create-course-btn`, `generate-guide-btn`, `add-milestone-btn`) against `Study_Window.py`'s `@on(Button.Pressed, ...)` decorators to confirm none of them dispatch to the legacy module (same shape as 16195's `add-topic-btn` finding).
4. Check for dynamic/reflective importers (`pkgutil`/`importlib` scans over `Event_Handlers`) that might load the package without a static `import` grep hit.
5. Based on the evidence, delete the dead surface — likely the whole `Event_Handlers/Study_Events/` package (both files), since preliminary grep shows zero external importers of the package itself, not just the table.
6. Grep docs/configs for dangling string references (`study_events`, `STUDY_BUTTON_HANDLERS`, handler names).
7. Re-run `pytest --collect-only Tests/` (zero errors) and the Study suites (`test_study_screen.py` + companions) to confirm no regression.
8. `ruff check`/`format` on touched files, hand-edit the task file's ACs/notes/status, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deleted the whole `Event_Handlers/Study_Events/` package (`study_events.py` +
`__init__.py`), not just `STUDY_BUTTON_HANDLERS`. Per-symbol sweep (grepping
`tldw_chatbook/` + `Tests/` for every hit outside the module itself) showed
every symbol was dead, not only the table:

| Symbol | Verdict | Evidence |
|---|---|---|
| `STUDY_BUTTON_HANDLERS` | dead | zero references outside its own definition site; `StudyWindow.on_button_pressed` (`Study_Window.py:946`) only special-cases 2 sidebar ids + a `view-` prefix, nothing else |
| `study_event_handler` (singleton) | dead | only used to build the (dead) table |
| `StudyEventHandler` (class) | dead | instantiated only as that singleton; package `__init__.py` re-exports it, but nothing outside the package imports `Study_Events` at all (`grep -rn "Study_Events" tldw_chatbook Tests` → only the `__init__.py` itself, plus a plain-text comment in `Study_Window.py`) |
| `StudyCardCreatedEvent` | dead | constructed only inside `handle_create_card` (itself unreachable — not in the table, no other dispatcher); no `@on`/handler for it anywhere; re-exported but package has no external importer |
| `StudyCardReviewedEvent` | dead (doubly) | never even constructed anywhere, not even within the dead module itself |
| `StudyTopicSelectedEvent` | dead | constructed only inside `handle_topic_selected`, which is wired to `Tree.NodeSelected` but no such handler exists for `#topic-tree` anywhere (`grep -n "NodeSelected"` finds only the method's own signature) |
| `handle_create_card`, `handle_topic_selected`, `handle_start_review` | dead | not even present in `STUDY_BUTTON_HANDLERS` — never reachable via any dispatch path, live or dead. (`Study_Window.py` has its own same-named `handle_create_card`/`handle_start_review` methods on `AnkiFlashcardsWidget`, wired via `@on(Button.Pressed, "#create-card-btn"/"#start-review-btn")` to `flashcards_controller` — unrelated, coincidentally-named, and unaffected by this deletion) |
| `handle_add_topic` | dead | its button (`#add-topic-btn`) was removed from `Study_Window.py` by task-16195 (merged as PR #1681, `19b0c1a03`, fast-forwarded onto this worktree first) |
| `handle_add_mindmap_child`, `handle_create_course`, `handle_generate_study_guide`, `handle_add_milestone` | dead | their buttons (`add-child-btn`, `create-course-btn`, `generate-guide-btn`, `add-milestone-btn`) still compose in `Study_Window.py` but have no `@on(Button.Pressed, "#...")` decorator anywhere and fall through `on_button_pressed`'s `view-`-prefix-only dispatch as silent no-ops — same shape 16195's review spot-checked for `add-child-btn`, now confirmed for all four |

No dynamic/reflective importers exist (`grep -rn "pkgutil|iter_modules|walk_packages"
tldw_chatbook` has zero hits touching `Event_Handlers`), so a static-import
grep sweep is conclusive.

Left a short addendum on the existing task-16195 comment in `Study_Window.py`
noting the whole module is now gone (it previously said only the table was
unreachable).

One dangling doc reference: `Docs/security/production-diagnostic-inventory.json`
carried a `tldw_chatbook/Event_Handlers/Study_Events/study_events.py` owner
row (23 diagnostic calls). Regenerating that file via
`scripts/check_persistent_diagnostic_inventory.py --write` pulled in a large
amount of **pre-existing, unrelated drift** (8 other files' call counts/digests
changed, one brand-new `trajectory_screen.py` owner appeared) — confirmed by
temporarily restoring `study_events.py` and the original inventory JSON and
re-running the checker, which still failed identically. That drift is a
known, separately-tracked recurring problem in this repo (task-1822, 2768,
3035, 3750, 14651, 15103, 15600, 15743 are all prior reconciliation passes)
and out of scope here, so I did **not** regenerate the whole file. Instead I
hand-removed just the `study_events.py` owner entry and decremented
`summary.owner_files` (491→490) and `summary.task_494_calls` (6948→6925) by
exactly its contribution, leaving every other (already-drifted) entry
untouched. `Tests/Architecture/test_persistent_diagnostic_inventory.py::
test_production_diagnostic_inventory_and_sink_topology_are_unchanged` was
already red before this change (verified) and stays red after it, for the
same pre-existing reasons — not a regression introduced here. The other 64
tests in that file still pass.

**Tests:** `pytest --collect-only Tests/` → 48173 collected, 0 errors, both
before and after (baseline unchanged). Study suites (`test_study_screen.py`,
`test_study_flashcards_screen.py`, `test_study_quizzes_screen.py`,
`test_study_dashboard.py`, `test_study_origin_navigation.py`,
`test_mount_io_off_pump.py`, `ChaChaNotesDB/test_study_functionality.py`) →
137 passed both before and after. `ruff check`/`format --check` on
`Study_Window.py` clean.

**Files changed:**
- Deleted `tldw_chatbook/Event_Handlers/Study_Events/study_events.py` and `__init__.py`
- `tldw_chatbook/UI/Study_Window.py` — updated the task-16195 comment
- `Docs/security/production-diagnostic-inventory.json` — removed the dangling `study_events.py` owner row + summary counts
<!-- SECTION:NOTES:END -->
