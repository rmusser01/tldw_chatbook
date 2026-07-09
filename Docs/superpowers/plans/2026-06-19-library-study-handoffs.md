# Library Study Handoffs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clarify Library-to-Study, Flashcards, and Quizzes handoffs so users know what source context is carried forward, what remains WIP, and how to recover when no source context exists.

**Architecture:** Keep Library as the source-preparation surface and preserve Study as the owner of sessions, flashcards, and quizzes. Add mode-specific Library copy and stable selectors around existing `StudyScopeContext` handoff behavior instead of duplicating Study internals.

**Tech Stack:** Python 3.12, Textual, existing `DestinationHarness` mounted UI tests, Backlog.md task tracking.

---

## ADR Check

ADR required: no
ADR path: N/A
Reason: this is a UI/handoff clarity slice under the existing TASK-89.1 Library content-hub contract. It does not change storage, sync policy, provider/runtime boundaries, service contracts, or persistence schemas.

## File Map

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
  - Add Study/Flashcards/Quizzes mode-specific purpose, source snapshot, WIP, and recovery copy.
  - Preserve existing `StudyScopeContext` generation and `open_study_screen(...)` routing.
- Modify: `Tests/UI/test_product_maturity_phase3_library_study_context.py`
  - Add mounted regressions for mode-specific handoff copy, empty recovery, and keyboard activation.
- Modify: `backlog/tasks/task-89.7 - Clarify-Library-study-flashcards-and-quizzes-handoffs.md`
  - Add implementation plan, checked acceptance criteria, and implementation notes when done.
- Add/Update: `Docs/superpowers/qa/library-study-handoffs/`
  - Store actual CDP/Textual-web screenshots once rendered UI is ready for approval.

## Task 1: Add Failing Regressions For Study Handoff Clarity

**Files:**
- Modify: `Tests/UI/test_product_maturity_phase3_library_study_context.py`

- [x] **Step 1: Write failing mounted tests**

Add tests that expect:
- `Study`, `Flashcards`, and `Quizzes` modes render source-context handoff copy.
- Empty Library source state explains that users can open Study globally but no Library source snapshot will be carried.
- Keyboard-only mode/action flow can activate the Flashcards handoff and pass `StudyScopeContext`.

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest -q Tests/UI/test_product_maturity_phase3_library_study_context.py --tb=short
```

Expected: new tests fail because the current Library modes still use generic knowledge workflow copy.

## Task 2: Implement Mode-Specific Library Study Panels

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`

- [x] **Step 1: Add a small helper for study handoff state**

Add a Library-screen helper that returns:
- mode label and destination section (`dashboard`, `flashcards`, `quizzes`)
- whether Library has source context
- copy for carried source snapshot
- copy for WIP/deferred work
- recovery copy for empty/no-source cases

- [x] **Step 2: Render mode-specific action copy**

In the source-action rendering path, branch for `study`, `flashcards`, and `quizzes` modes before the generic knowledge workflow. Keep the existing button IDs:
- `#library-open-study`
- `#library-open-flashcards`
- `#library-open-quizzes`

Add stable Static IDs for regression and QA:
- `#library-study-handoff-purpose`
- `#library-study-handoff-context`
- `#library-study-handoff-wip`
- `#library-study-handoff-recovery`

- [x] **Step 3: Keep handoff routing unchanged**

Do not change `_source_study_context()` or `_open_study_section(...)` except for safe copy/readiness support if required by tests.

## Task 3: Verify And Document

**Files:**
- Modify: `backlog/tasks/task-89.7 - Clarify-Library-study-flashcards-and-quizzes-handoffs.md`
- Add/Update: `Docs/superpowers/qa/library-study-handoffs/`

- [x] **Step 1: Run focused tests**

Run:

```bash
python -m pytest -q Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_library_content_hub.py --tb=short
```

Expected: all pass.

- [x] **Step 2: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [x] **Step 3: Capture actual rendered screenshots**

Use Textual-web/CDP against the running app and capture:
- Study mode handoff panel with seeded Library sources.
- Flashcards mode handoff panel with seeded Library sources.
- Quizzes mode handoff panel with empty/no-source recovery.

Store captures under `Docs/superpowers/qa/library-study-handoffs/` and get user approval before PR.

Evidence captured and approved:
- `Docs/superpowers/qa/library-study-handoffs/library-study-handoff-seeded-cdp-2026-06-19.png`
- `Docs/superpowers/qa/library-study-handoffs/library-flashcards-handoff-seeded-cdp-2026-06-19.png`
- `Docs/superpowers/qa/library-study-handoffs/library-quizzes-handoff-empty-cdp-2026-06-19.png`

- [x] **Step 4: Update Backlog task**

Check completed acceptance criteria, add implementation notes, and move TASK-89.7 to Done only after tests and screenshot approval.

- [ ] **Step 5: Commit**

Commit the test/code/docs changes in a small reviewable commit.
