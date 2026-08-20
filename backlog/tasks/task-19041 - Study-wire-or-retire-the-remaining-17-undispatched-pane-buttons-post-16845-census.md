---
id: TASK-19041
title: >-
  Study: wire or retire the remaining 17 undispatched pane buttons
  (post-16845 census)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-20 08:40'
labels:
  - ui
  - dead-code
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16845 removed four undispatched Study buttons (add-child / create-course /
generate-guide / add-milestone) plus the Course form, and explicitly deferred the
rest of the surface as "materially larger" (see its Implementation Notes'
out-of-scope list). Fresh census at dev `1bf7f234e`: `UI/Study_Window.py` composes
48 buttons (42 literal ids + 6 f-string `review-rating-*`), of which 24 distinct
ids reach a handler (21 `@on(Button.Pressed, ...)` targets incl. the six rating
ids, plus `on_button_pressed`'s `study-back-to-workspace-button`,
`study-switch-global-button`, and the seven `view-*` sidebar ids). That leaves
**18 composed button instances across 17 distinct ids with no dispatch anywhere**
(whole-tree grep per id returns only `Study_Window.py` and tests):

`add-sibling-btn`, `delete-node-btn`, `edit-node-btn`, `import-notes-btn`,
`export-md-btn` (composed TWICE — Mindmaps pane :451 and Course pane :512, a
duplicate-id wrinkle of its own), `generate-mindmap-btn`, `add-module-btn`,
`export-pdf-btn`, `export-scorm-btn`, `add-concept-btn`,
`generate-questions-btn`, `save-guide-btn`, `mark-complete-btn`,
`set-dependencies-btn`, `import-course-btn`, `export-path-btn`,
`generate-suggestions-btn`.

`on_button_pressed` early-returns for every one of them, so each press is a
silent no-op — the exact UX shape 16195's review called worse than "does
nothing" implies. 16845's per-pane evidence already established the backing is
placeholder-grade (write-only or nonexistent schema; static trees nothing
populates), so per the owner's stability ruling the default expectation is
honest removal / honest empty state per pane, not speculative wiring to
write-only sinks. No existing backlog task covers these ids (grepped
backlog/tasks at dev).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No composed Study control silently swallows a press: each of the 17 ids either reaches a real handler or is removed with per-affordance evidence (16195/16845 pattern, including controls that exist solely to feed a removed button)
- [x] #2 Wire-vs-remove is decided per pane on schema/service evidence, preferring durable removal or an honest empty state over wiring to write-only sinks (owner ruling: stability over quick wins)
- [x] #3 The duplicate `export-md-btn` id no longer composes twice
- [x] #4 Study suites stay green and pinning tests forbid removed affordances from returning
<!-- AC:END -->

## Implementation Plan

1. Re-census at branch base (origin/dev `25500ad87`): confirm all 17 ids compose in
   `UI/Study_Window.py` with zero dispatch anywhere (whole-tree grep per id), and
   `export-md-btn` composes twice.
2. Evidence pass per pane, extending 16845's: grep `ChaChaNotes_DB.py` for
   mindmap/course/module/guide/concept/milestone/dependency/learning-path methods and
   `Study_Interop/` for any service backing; record wire-vs-remove per id.
3. Decide per pane on that evidence (expectation per the owner stability ruling:
   removal + honest empty state, mirroring 16845's Structured Learning treatment; the
   mindmap SUBSYSTEM files are task-19042's and stay untouched — only the
   Study_Window.py pane chrome is in scope).
4. Update the four 16845 pin tests in `Tests/UI/test_study_screen.py` to forbid every
   removed id and assert each pane's empty-state notice; run them against the
   unmodified `Study_Window.py` first to show them red for the right reason
   (born-red), keeping the always-true control assertions passing.
5. Implement the removals/pane rewrites in `UI/Study_Window.py` with removal comments
   (16195 pattern) + dead-CSS cleanup; drop `Tests/UI/test_study_guide_topic_select.py`
   if its subject control (`#guide-topic-select`) is removed with its pane.
6. Green the updated pins; one representative mutation (Edit-based re-add of a removed
   id, run pin red, Edit-based restore) to prove the pins bite.
7. Targeted suites (`test_study_screen.py`, `test_study_flashcards_screen.py`,
   `test_study_quizzes_screen.py`, `test_study_dashboard.py`,
   `test_study_origin_navigation.py`, `test_mount_io_off_pump.py`,
   `Tests/ChaChaNotesDB/test_study_functionality.py`) + repo-wide `--collect-only -q`
   sweep; ruff check/format on touched files.
8. Docs check: no `Docs/User_Guide/` page documents these panes (Study appears only as
   flashcards/quizzes in index/home/library) — record the grep, no page to stamp.
9. Hand-edit this task file (5-digit id: no backlog CLI): tick ACs, Implementation
   Notes, status Done; single commit on `task/19041-burn`.

## Implementation Notes

**Decision: REMOVE all 17 ids (18 composed instances) and replace each of the
four placeholder panes — Mindmaps, Course Creation, Study Guide, Learning Map —
with an honest empty-state `Static`, mirroring 16845's Structured Learning
treatment.** Re-census at branch base `25500ad87` confirmed every id composes
only in `UI/Study_Window.py` with zero dispatch anywhere (whole-tree grep per
id: hits only compose + tests), and no pane had backing worth wiring to:

| Pane | Ids removed | Evidence |
|---|---|---|
| Mindmaps | `add-sibling-btn`, `delete-node-btn`, `edit-node-btn`, `import-notes-btn`, `export-md-btn`, `generate-mindmap-btn` | `create_mindmap`/`add_mindmap_node` are the only mindmap methods in `ChaChaNotes_DB.py` (write-only, no read/list); `#mindmap-tree` was a static "Root Topic" skeleton with no population code — nothing added/edited/imported/generated could ever display or export. `Tools/Mind_Map/mindmap_integration.py::create_from_notes` exists but the whole subsystem is production-orphaned and is task-19042's concurrent wire-or-retire scope (retirement preferred) — not a wiring target. |
| Course Creation | `add-module-btn`, `export-pdf-btn`, `export-md-btn` (the duplicate instance), `export-scorm-btn` | No `course`/`module` concept exists in any schema or `Study_Interop/` service (grep: zero method hits); `#module-list` never populated; SCORM has **zero** hits anywhere in `tldw_chatbook/` outside this pane — no exportable course exists in any format. |
| Study Guide | `add-concept-btn`, `generate-questions-btn`, `save-guide-btn` | No guide/concept table or service exists anywhere; `#save-guide-btn` had no destination, `#add-concept-btn` fed an in-session list nothing persists or reads, `#generate-questions-btn` had no generation service (`study_scope_service.py`'s "suggestion" methods are flashcard-tag suggestions, a different feature). |
| Learning Map | `mark-complete-btn`, `set-dependencies-btn`, `import-course-btn`, `export-path-btn`, `generate-suggestions-btn` | Only conceivable sink for mark-complete is the write-only `update_topic_progress` (16195's finding — nothing reads it back); no dependency/suggestion concept exists anywhere; no course to import; `learning_paths` write-only so nothing to export; `#learning-map-tree` and `#overall-progress`/`#current-topic` were statics nothing populates/updates. |

**Feeder/chrome removed with the buttons (16195 "whole affordance" pattern,
per-pane coherence):** Mindmaps `#node-text` + `#mindmap-tree`; Course
`#module-name` + `#module-list`; Study Guide `#guide-topic-select` (one static
option, `.value` consumer-less per the TASK-16841 sweep), `#guide-title`,
`#guide-content`, `#concept-input`, `#key-concepts-list`,
`#practice-questions-list`; Learning Map `#learning-map-tree`,
`#overall-progress`, `#current-topic`. Each pane keeps its section title and
gains an empty-state notice stating plainly the feature isn't built
(`#mindmaps-empty-state`, `#course-creation-empty-state`,
`#study-guide-empty-state`, `#learning-map-empty-state`). Matching dead-CSS
cleanup in each widget's DEFAULT_CSS; unused `Tree` import dropped.

**AC#3:** both `export-md-btn` instances are gone (the two never co-existed in
the DOM — panes are alternate views — so the duplicate was a source-level
hazard; the new source-level pin asserts zero `id="export-md-btn"` occurrences).

**Boundary honored:** `Tools/Mind_Map/`, `UI/Widgets/MindmapViewer.py`,
optional-dep plumbing, and ChaChaNotes mindmap accessors untouched
(task-19042's concurrent scope). The Study pane never composed those symbols.

**Pinning tests (born red):** in `Tests/UI/test_study_screen.py` — a new
source-level pin (`test_removed_study_pane_affordances_do_not_return_in_source`,
regex on `id="..."` assignments over a 38-id tuple covering 16845's four ids +
this task's 17 + all feeder/chrome ids) plus the four 16845 pane pins rewritten
to their 19041 form (`test_*_pane_composes_only_the_honest_empty_state`:
navigate via the production sidebar, assert every removed id absent and the
empty-state notice present). Run against the UNMODIFIED product file first:
all 5 failed for exactly the right reasons (source pin listed all 30
then-present ids; each pane pin failed on its first still-composing id) while
the structured-learning control pin passed. Green after the change. Mutation
check: re-added `export-md-btn` to the Course pane via Edit — both the source
pin and the pane pin went red naming the exact offender — then Edit-restored.
`Tests/UI/test_study_guide_topic_select.py` (TASK-16841's label/value-order
pin) was deleted: its subject control `#guide-topic-select` no longer composes,
and the new pins forbid the id's return.

**Tests:** `test_study_screen.py` 24 passed (18 pre-existing + 5 new/rewritten
pins + structured-learning pin). Full family (`test_study_screen.py`,
`test_study_flashcards_screen.py`, `test_study_quizzes_screen.py`,
`test_study_dashboard.py`, `test_study_origin_navigation.py`,
`test_mount_io_off_pump.py`, `Tests/ChaChaNotesDB/test_study_functionality.py`,
`Tests/Architecture/test_backwards_select_option_guard.py`,
`test_product_maturity_phase3_library_study_context.py`,
`test_product_maturity_phase3_source_study_generation.py`) → **162 passed**.
Repo-wide `--collect-only -q` → **51,470 collected, 0 errors**. `ruff check` +
`format --check` clean on both touched files. Residue grep for every removed
id/affordance across `tldw_chatbook/`, `Tests/`, `Docs/` → hits only in
`Study_Window.py` removal comments and the pins.

**Docs:** no `Docs/User_Guide/` page documents these panes — Study appears only
as flashcards/quizzes hand-offs (index.md, home.md, library.md; grep for
mindmap/course/learning map/structured learning across `Docs/User_Guide/*.md`
returns nothing) — same finding as 16195/16845, so no page to update or stamp.

**Known residue for the owner (out of scope):** `UI/Screens/study_screen.py`'s
sidebar still offers Guides/Mindmaps/Course/Map entries whose tooltips promise
generation/exploration ("Generate or open study guides from your material.",
etc.). The buttons are live and now land on honest empty states, but the copy
overpromises; deciding whether to hide unbuilt sections or soften the tooltips
is a screen-IA call, not a dead-control repair.

**Files changed:** `tldw_chatbook/UI/Study_Window.py` (four pane rewrites +
CSS + import cleanup), `Tests/UI/test_study_screen.py` (source pin + four
rewritten pane pins), `Tests/UI/test_study_guide_topic_select.py` (deleted),
this task file.
