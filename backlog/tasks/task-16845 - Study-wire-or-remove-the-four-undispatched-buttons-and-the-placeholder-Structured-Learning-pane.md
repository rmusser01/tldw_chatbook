---
id: TASK-16845
title: 'Study: wire or remove the four undispatched buttons and the placeholder Structured Learning pane'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
labels:
  - ui
  - dead-code
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16195 (PR #1681) removed the orphaned "Add Topic" affordance and TASK-16196
(PR #1688) deleted the legacy Study event-handler module whose table was the only thing
that ever named these buttons. Both reviews confirmed — and it still holds at dev
`ee741cf10` — that **four more Study buttons compose live with no handler anywhere**:

- `UI/Study_Window.py:440` — `Button("Add Child", id="add-child-btn")`
- `UI/Study_Window.py:524` — `Button("Create Course", id="create-course-btn")`
- `UI/Study_Window.py:612` — `Button("Generate from Topic", id="generate-guide-btn")`
- `UI/Study_Window.py:671` — `Button("Add Milestone", id="add-milestone-btn")`

Zero `@on(Button.Pressed, ...)` decorators reference any of the four (re-grepped at
HEAD), and `StudyWindow.on_button_pressed` early-returns unless the id is one of two
sidebar ids or starts with `view-` — so each press is a **silent no-op**: a user can fill
in the adjacent inputs, click, and get no signal at all. Same shape as the removed
add-topic button, which 16195's review called worse UX than "does nothing" implies.

The same review raised the coherence question this task should settle alongside: the
Structured Learning pane's residual chrome (`#topic-tree` rooted at a static "Learning
Paths" node that can never gain children; `#topic-content`, a disabled TextArea whose
placeholder promises "Select a topic from the tree..." with nothing selectable and topics
write-only in the DB — no row-level read method exists) reads as broken rather than
intentionally empty. Per affordance: wire it to the live Study rebuild
(`UI/Study_Modules/` is flashcards + quizzes only today), remove it with 16195's
per-affordance evidence pattern, or replace the pane with honest empty-state copy /
gate the section until its backing feature exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 No composed Study control silently swallows a press: every remaining button has a real handler or is removed (per-button evidence, 16195-style)
- [x] #2 The Structured Learning pane either gains a real read path or presents an honest, intentional empty state (no dead tree + disabled content pane promising interaction)
- [x] #3 Study suites stay green and the pinning tests forbid the removed affordances from returning
<!-- AC:END -->

## Implementation Plan

1. Evidence pass per button: grep the whole tree for `mindmaps`/`mindmap_nodes`,
   `learning_paths`/`topics`, `course`, `study_guide`/`guide`, `milestone` schema
   tables and any read (list/get) DB methods, plus `Study_Interop/*` service methods,
   to establish whether each of the four panes (Mindmaps, Course Creation, Study
   Guide, Learning Map) has any live backing beyond a write-only sink.
2. Confirm (re-grep) all four target buttons still have zero `@on(Button.Pressed, ...)`
   dispatch anywhere and that the legacy `Event_Handlers/Study_Events` module stays
   deleted (task-16196 baseline).
3. Decide wire vs remove per button on that evidence, and separately decide the
   Structured Learning pane's coherence treatment (honest empty state vs real read
   path vs gating), recording the reasoning.
4. Write/extend born-red pinning tests in `Tests/UI/test_study_screen.py` for each
   pane, run them at HEAD to confirm red for the right reason (control assertions
   still pass, target assertions fail).
5. Implement the chosen removals/pane rewrite in `tldw_chatbook/UI/Study_Window.py`
   with explanatory comments at each removal site (matching the task-16195 pattern).
6. Re-run the new/updated tests (green) plus the full Study suite family used by
   16195/16196 (`test_study_screen.py`, `test_study_flashcards_screen.py`,
   `test_study_quizzes_screen.py`, `test_study_dashboard.py`,
   `test_study_origin_navigation.py`, `test_mount_io_off_pump.py`,
   `ChaChaNotesDB/test_study_functionality.py`).
7. ruff check/format on touched files; hand-edit the task file's ACs/notes/status
   (5-digit IDs mangle under the CLI); commit locally (no push/PR/merge).

## Implementation Notes

**Decision: REMOVE all four buttons.** None had schema/service support strong
enough to justify wiring, and each pane's surrounding chrome (tree/select
population, save destinations) is dead in the same shape 16195 already
documented for `topics`. Per-button evidence:

| Button | Verdict | Evidence |
|---|---|---|
| `add-child-btn` (Mindmaps, "Add Child") | REMOVE | `ChaChaNotes_DB.create_mindmap`/`add_mindmap_node` exist but are write-only — grepping the whole file for `get_mindmap`/`list_mindmap*` returns nothing, and `#mindmap-tree` composes with a static "Root Topic" root and no population code. A written node could never be displayed, in any session (same shape as 16195's topics finding). |
| `create-course-btn` (Course Creation) | REMOVE | No `course`/`courses` table exists **anywhere** in `ChaChaNotes_DB.py` (zero grep hits) — unlike the other three, there isn't even a write-only sink to point the button at. The "course" hits elsewhere in the tree (`study_screen.py`'s sidebar dispatch, `study_scope_models.py`, `Stats/user_statistics.py`, `ChatbookTemplatesWindow.py`) are unrelated: pure UI navigation to the placeholder pane, an unrelated stat-category label, and unrelated placeholder copy respectively — none of them back this button. |
| `generate-guide-btn` (Study Guide, "Generate from Topic") | REMOVE | `#guide-topic-select` is hard-coded to a single static `("new", "New Topic")` option with zero code populating it from the `topics` table (also write-only, per 16195) — there is no topic the button could ever generate from, and no `study_guide`/`guide` table exists to save output to either. A coincidentally-named, architecturally unrelated "study guide" feature exists in `Chat/document_generator.py` (`generate_study_guide`, wired via `Widgets/document_generation_modal.py::handle_study_guide`) — it generates from **conversation** context, not topics, lives on a different screen, and is not a viable redirect target without a feature build (same "unrelated, coincidentally-named" shape 16196 found for `handle_create_card`). |
| `add-milestone-btn` (Learning Map) | REMOVE | No `milestone` concept exists anywhere in the schema (zero grep hits in `ChaChaNotes_DB.py`); `#learning-map-tree` composes with a static "Learning Path" root and no population code, and `#overall-progress`/`#current-topic` are hard-coded statics nothing ever updates. Same placeholder-grade shape as the topic tree. |

**Removal scope per button** (mirroring 16195's "remove the whole affordance,
not just the button" when other controls exist solely to feed it):
- Mindmaps: removed only the `Button` line — `#node-text` Input stays because
  the separate, equally-undispatched `#add-sibling-btn` still uses it.
- Course Creation: removed the entire "Course Details" form (title/
  description/level/prerequisites), since none of those four fields fed
  any other remaining button — leaving them would have produced four
  orphaned inputs with no possible destination at all, which 16195's review
  already called worse than a dead button. Course Modules and Export
  Options are separate sections with their own (also undispatched, out of
  scope) buttons and were left untouched.
- Study Guide / Learning Map: removed only the `Button` line in both cases —
  their neighboring fields/statics are either shared with a remaining
  button or already general-purpose/unpopulated regardless.

**Structured Learning pane coherence (AC #2): replaced the dead chrome with
an honest empty-state notice**, not a real read path. Building list/get
methods for `learning_paths`/`topics` plus tree-population plumbing would be
a feature build, not a wiring fix — the same call 16195 made for the
add-topic write path. `#topic-tree` (static "Learning Paths" root, no
population code, no `Tree.NodeSelected` handler anywhere since 16196 deleted
the legacy module that held one) and the disabled `#topic-content` TextArea
(placeholder promising "Select a topic from the tree to view content..."
with nothing ever selectable) were removed and replaced with a single
`Static` (`#structured-learning-empty-state`) stating plainly that
Structured Learning has no browsing UI yet.

**Explicitly out of scope** (same discipline 16195/16196 applied): every
*other* button in the Mindmaps/Course Creation/Study Guide/Learning Map
panes — `add-sibling-btn`, `delete-node-btn`, `edit-node-btn`,
`import-notes-btn`, `export-md-btn` (×2, one per pane), `generate-mindmap-btn`,
`add-module-btn`, `export-pdf-btn`, `export-scorm-btn`, `add-concept-btn`,
`generate-questions-btn`, `save-guide-btn`, `mark-complete-btn`,
`set-dependencies-btn`, `import-course-btn`, `export-path-btn`,
`generate-suggestions-btn` — is equally undispatched (confirmed while
reading each pane's `compose()` in full: none of these classes define any
`@on`/`on_button_pressed` method beyond `compose()` itself) but was never
part of the legacy `STUDY_BUTTON_HANDLERS` table this task's four buttons
came from, and is a materially larger surface (an entire-pane rebuild, not
a per-button wire/remove call). Left untouched, same as 16195/16196 deferred
them.

**Pinning tests (born red):** five tests in `Tests/UI/test_study_screen.py`,
each mounting the production app via the existing `_build_full_study_app`
harness and navigating to the relevant pane via its `#view-*-btn` sidebar
button:
- `test_structured_learning_pane_composes_no_orphaned_add_topic_controls`
  (extended) — asserts `#topic-tree`/`#topic-content` are gone and
  `#structured-learning-empty-state` renders the honest notice.
- `test_mindmaps_pane_composes_no_orphaned_add_child_button` (new)
- `test_course_creation_pane_composes_no_orphaned_create_course_form` (new)
- `test_study_guide_pane_composes_no_orphaned_generate_guide_button` (new)
- `test_learning_map_pane_composes_no_orphaned_add_milestone_button` (new)

Each also asserts the relevant out-of-scope sibling controls still compose
(e.g. `#add-sibling-btn`, `#add-module-btn`, `#save-guide-btn`,
`#mark-complete-btn`), so the pinning is precise about what was and wasn't
removed. Ran all five at HEAD before the product change: all 5 failed for
the expected reason (target control still present / empty-state Static
absent). After the change: all 5 pass.

**Tests:** `test_study_screen.py` 23 passed (18 pre-existing + 5 new/extended).
Full 16195/16196 suite family (`test_study_screen.py`,
`test_study_flashcards_screen.py`, `test_study_quizzes_screen.py`,
`test_study_dashboard.py`, `test_study_origin_navigation.py`,
`test_mount_io_off_pump.py`, `Tests/ChaChaNotesDB/test_study_functionality.py`)
→ 141 passed. `Tests/UI` collect-only sweep → 13,067 collected, 0 errors.
`ruff check`/`format --check` clean on both touched files. Grepped
`tldw_chatbook/` + `Tests/` for the four removed ids and the removed course
form's ids (`course-title`, `course-description`, `course-level`,
`course-prerequisites`, `topic-content`) — no remaining references outside
`Study_Window.py` (removal comments) and `Tests/UI/test_study_screen.py`
(absence assertions). No `Docs/User_Guide` page exists for Study (same as
16195's finding), so no docs update needed.

**Files changed:** `tldw_chatbook/UI/Study_Window.py` (four button removals
+ Structured Learning pane rewrite + matching dead-CSS cleanup:
`.topic-tree`/`.topic-content` → `.structured-learning-empty-state`,
`.course-form` and `.course-description` rules dropped), `Tests/UI/test_study_screen.py`
(one extended test + four new tests).
