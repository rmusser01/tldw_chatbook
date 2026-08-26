---
id: TASK-16195
title: 'Study add-topic button has no handler at HEAD'
status: Done
assignee: []
created_date: '2026-08-14 03:05'
labels:
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`#add-topic-btn` is composed in the Study surface but no handler is wired to it at current dev HEAD — pressing it does nothing. Found while verifying TASK-15471's dead-code justification for the legacy `study_events.py` handler table (which DID contain an add-topic handler; the Study rebuild moved flashcards to `Study_Modules/flashcards_handler.py` but the add-topic wiring was left behind). Decide the intended behavior (restore an add-topic flow in the current Study modules, or remove the dead button) and implement it. Related: TASK-16196 deletes the orphaned legacy handler module. Surfaced during TASK-15471 (per-click I/O off-loop, PR #1625 merged `172ada448`) and its concurrency review; evidence in the session review record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The add-topic button either performs its intended action end-to-end or is removed from compose
- [x] #2 A test pins whichever behavior is chosen
- [x] #3 The decision is recorded in the notes
<!-- AC:END -->

## Implementation Plan

1. Evidence pass at HEAD: who dispatches `STUDY_BUTTON_HANDLERS`; who reads the `topics` table (DB methods, Study_Interop services, dashboard widgets); does the legacy `handle_add_topic` payload still map onto `ChaChaNotes_DB.create_topic`.
2. Decide restore-vs-remove on that evidence (recorded in Implementation Notes).
3. Born-red pinning test in `Tests/UI/test_study_screen.py` using the existing `_build_full_study_app` harness: the Structured Learning pane renders its topic tree but composes NO add-topic affordance (`#add-topic-btn`, `#new-topic-title`). Run at HEAD to show it red.
4. Implement the chosen behavior (removal of the dead "Add New Topic" row from `StructuredLearningWidget.compose`), re-run test green plus the surrounding Study UI test files.
5. ruff check + format on touched files; hand-edit task file to Done with notes (backlog CLI mangles five-digit IDs).

## Implementation Notes

**Decision: REMOVE the dead "Add New Topic" affordance** (Label + `#new-topic-title` Input + `#add-topic-btn` Button) from `StructuredLearningWidget.compose` in `tldw_chatbook/UI/Study_Window.py`, with an explanatory comment left at the removal site.

**Evidence chain (why removal, not restore):**

1. **No dispatcher at HEAD.** The legacy `STUDY_BUTTON_HANDLERS` table in `Event_Handlers/Study_Events/study_events.py` is imported only by its own package `__init__.py`; nothing in `tldw_chatbook/` or `Tests/` imports the `Study_Events` package. The handler is unreachable.
2. **Topics are a write-only concept in the app.** `ChaChaNotes_DB` has `create_topic` and `update_topic_progress` but NO read/list method for topics anywhere; `Study_Interop/local_study_service.py` covers decks/flashcards/templates only, `server_study_service.py` never touches the topics table, and the Study dashboard has no topic surface. A row the button created could never be displayed, in any session.
3. **The Structured Learning pane is placeholder-grade at HEAD.** `#topic-tree` composes empty ("Learning Paths" root) with no population code; `#topic-content` is a disabled TextArea with placeholder text; the companion `Tree.NodeSelected` handler lives in the same orphaned legacy module. Even when the legacy wiring was live, the tree add was in-session only — restart showed an empty tree.
4. **The legacy handler's payload doesn't map onto the live schema.** Topics belong to a learning path (`path_id` FK into `learning_paths`); the button had no path context (would insert NULL `path_id`) and passed `created_by`/`last_modified_by` keys that `create_topic` ignores in favor of `client_id`.
5. **The Study rebuild deliberately scoped its module architecture (Study_Modules controllers + Study_Interop scope services) to flashcards + quizzes.** The live "topic" concepts elsewhere (quiz `focus_topics`/`selected_topic_ids`, conversation `topic_label`) are different features, not the `topics` table.

Restoring would have required building an entire read/display path (DB list method, scope-service plumbing, tree population) for a mock pane — a feature build, not a wiring fix — and porting only the write would ship a silent write-only data sink that fakes success. Removal is the durable choice (per the stability-over-quick-wins ruling).

**Pinning test (born red):** `Tests/UI/test_study_screen.py::test_structured_learning_pane_composes_no_orphaned_add_topic_controls` mounts the production app on the Study screen via the existing `_build_full_study_app` harness, asserts `#topic-tree` still composes, and asserts `#add-topic-btn`/`#new-topic-title` are absent. Run at HEAD before the product change it failed on exactly the button's presence (`assert not study_window.query("#add-topic-btn")` → `AssertionError`); green after the removal.

**Tests run:** `test_study_screen.py` 19 passed; `test_study_flashcards_screen.py` + `test_study_quizzes_screen.py` + `test_study_dashboard.py` + `test_study_origin_navigation.py` + `test_mount_io_off_pump.py` 56 passed; `test_product_maturity_phase3_library_study_context.py` + `test_product_maturity_phase3_source_study_generation.py` + `Tests/ChaChaNotesDB/test_study_functionality.py` 78 passed; `Tests/UI` collect-only sweep 12,531 collected clean. ruff check + format clean on touched files. No user-guide page documents the control (none exists for Study), so no docs update.

**Coordination with task-16196 (legacy module deletion):** nothing needs preserving. With removal chosen, `handle_add_topic`, `handle_topic_selected`, `StudyTopicSelectedEvent`, and the whole `STUDY_BUTTON_HANDLERS` table can be deleted with the module. Note for the record: the table's other entries (`add-child-btn`, `create-course-btn`, `generate-guide-btn`, `add-milestone-btn`) are equally undispatched at HEAD — those buttons remain composed in their (also placeholder) panes and are out of this task's scope.

**Files changed:** `tldw_chatbook/UI/Study_Window.py` (removal), `Tests/UI/test_study_screen.py` (pinning test + `Tree` import).
