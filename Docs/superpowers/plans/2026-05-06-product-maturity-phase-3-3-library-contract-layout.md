# Product Maturity Phase 3.3 Library Contract Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Library destination visibly follow the approved Phase 3.0 layout contract while preserving existing Library actions and source snapshot behavior.

**Architecture:** Keep the existing `LibraryScreen` data/service seams and only reshape the composed shell into a contract-aligned header, mode bar, source browser, detail, and inspector. Add mounted regressions that verify the contract essentials across terminal sizes, then record QA and roadmap evidence under the product-maturity Phase 3 tracker.

**Tech Stack:** Python 3.12, Textual, pytest, Backlog.md.

---

### Task 1: Add Library Contract Layout Regression

**Files:**
- Create: `Tests/UI/test_product_maturity_phase3_library_contract_layout.py`
- Read: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- Read: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Write failing mounted tests**

Add tests that mount Library through `DestinationHarness`, seed local source services, and assert:

- `#library-mode-bar` exists and exposes Sources, Search/RAG, Import/Export, Workspaces, Study, Flashcards, and Quizzes.
- `#library-source-browser`, `#library-source-detail`, and `#library-source-inspector` exist.
- Existing action selectors remain present: `#library-open-notes`, `#library-open-media`, `#library-open-conversations`, `#library-open-import-export`, `#library-open-search`, `#library-open-study`, `#library-open-flashcards`, `#library-open-quizzes`, and `#library-use-in-console`.
- Compact, default, and large terminal runs retain the contract essentials.

- [ ] **Step 2: Run the focused red tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py --tb=short
```

Expected before implementation: fail because the Library contract layout selectors do not exist yet.

### Task 2: Reshape LibraryScreen Composition

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: `Tests/UI/test_product_maturity_phase3_library_contract_layout.py`
- Test: `Tests/UI/test_product_maturity_phase3_knowledge_entry.py`
- Test: `Tests/UI/test_product_maturity_phase3_library_study_context.py`
- Test: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Implement the minimal layout structure**

In `LibraryScreen.compose_content()`, keep existing snapshot logic and action handlers, but compose:

- destination header and status row.
- `#library-mode-bar` with mode labels/actions.
- `#library-contract-grid`.
- `#library-source-browser` for Notes, Media, Conversations, and Workspaces scope cue.
- `#library-source-detail` for source snapshot loading/error/empty/summary records.
- `#library-source-inspector` for authority, Search/RAG, Study, and Console handoff actions.

- [ ] **Step 2: Run focused green tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_destination_shells.py --tb=short
```

Expected after implementation: pass with only known dependency warnings.

### Task 3: Record Phase 3.3 Evidence And Tracking

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-3-library-contract-layout.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
- Modify: `backlog/tasks/task-10.3 - Product-Maturity-Phase-3.3-Library-Contract-Layout-Shell.md`
- Test: `Tests/UI/test_product_maturity_phase3_library_contract_layout.py`

- [ ] **Step 1: Add evidence assertions**

Extend the Phase 3.3 test file to verify QA evidence, roadmap entries, and Backlog task hygiene.

- [ ] **Step 2: Run red evidence tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py::test_phase_3_3_library_contract_layout_evidence_is_tracked --tb=short
```

Expected before docs/tracker updates: fail because evidence and tracker entries are missing.

- [ ] **Step 3: Add QA evidence and tracking updates**

Record the compact/default/large walkthrough, changed seams, focused verification, P0/P1 result, and residual risks.

- [ ] **Step 4: Complete Backlog task hygiene**

Check all `TASK-10.3` acceptance criteria, add implementation notes, and append a concise parent `TASK-10` note that Phase 3.3 closed the Library layout shell gate while deeper source-selected generation and Workspaces/Collections flows remain.

- [ ] **Step 5: Run final verification**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_destination_shells.py Tests/UI/test_product_maturity_phase3_layout_contracts.py --tb=short
git diff --check
```

Expected: all tests pass and diff check is clean.
