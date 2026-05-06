# Phase 3.1 Library Study Entry

Date: 2026-05-06
Task: TASK-10.1
Branch: codex/product-maturity-phase3-1-knowledge-entry
Workflow: Library -> Study Dashboard / Flashcards / Quizzes

## What Was Verified

Phase 3.1 starts Knowledge and Study workflow depth by making Study, Flashcards, and Quizzes visibly reachable from Library instead of relying on hidden routes or prior knowledge.

Verified contract:

- Library exposes a `Knowledge workflow` section near source, import/export, and Search/RAG entry points.
- Library shows dedicated `Study Dashboard`, `Flashcards`, and `Quizzes` controls with beginner-readable tooltips.
- Flashcards and Quizzes preserve the requested Study section through `open_study_screen(initial_section=...)`.
- Study consumes the pending initial section and clears it so later Study opens do not unexpectedly reuse stale routing state.

## Automated Evidence

Initial red run:

```bash
.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase3_knowledge_entry.py -q
```

Result:

- `5 failed, 3 warnings in 10.94s`.
- Failures were expected: Library had no Knowledge workflow controls, Study did not consume a pending initial section, `open_study_screen` did not accept `initial_section`, and Phase 3.1 evidence did not exist yet.

Focused behavior verification:

```bash
.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase3_knowledge_entry.py::test_library_surfaces_study_workflow_entry_points Tests/UI/test_product_maturity_phase3_knowledge_entry.py::test_library_study_entry_buttons_preserve_requested_section Tests/UI/test_product_maturity_phase3_knowledge_entry.py::test_study_screen_consumes_pending_initial_section Tests/UI/test_product_maturity_phase3_knowledge_entry.py::test_tldwcli_open_study_screen_accepts_initial_section -q
```

Result:

- `4 passed, 1 warning in 8.05s`.

Closeout tracking verification:

```bash
.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase3_knowledge_entry.py -q
```

Result:

- `6 passed, 1 warning in 6.51s`.

Broader adjacent verification:

```bash
.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_destination_shells.py::test_library_exposes_source_sections_and_import_export_boundary Tests/UI/test_destination_shells.py::test_destination_action_buttons_emit_compatibility_routes Tests/UI/test_study_dashboard.py -q
```

Result:

- `22 passed, 1 warning in 16.95s`.

## Manual Walkthrough

Mounted Library walkthrough verified the entry points are visible even when Library source services are unavailable. This matters for first-time and empty/error states: the user can still discover Study, Flashcards, and Quizzes while source population remains blocked or empty.

## Visual/Focus Notes

- The new controls are grouped under `Knowledge workflow`, which distinguishes study derivatives from source browsing and Search/RAG.
- Buttons retain Textual focusable controls and explicit tooltips rather than adding passive copy only.
- The first slice intentionally keeps the Study destination under Library instead of adding a new top-level nav item, preserving the current shell IA while making flashcards and quizzes visible in the product model.

## Defects Found

No P0/P1 defects found.

## Residual Risk

- This slice routes to Study, Flashcards, and Quizzes but does not yet generate flashcards or quizzes directly from selected Library sources.
- Workspaces and Collections remain later Phase 3 gates.
- Import/Export progress and recovery are still covered only by existing entry-point routing and need deeper Phase 3 validation.

## Exit Decision

Pass for Phase 3.1. Library now exposes the first visible Knowledge and Study workflow entry points and preserves the requested Study section for Flashcards and Quizzes.
