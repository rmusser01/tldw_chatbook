# Product Maturity Phase 3.7: Source Study-Pack Completion Reuse

Date: 2026-05-07
Task: `TASK-10.7`
Status: verified

## Scope

Verify the next source-selected Study generation gate after Phase 3.4:

`Library-selected sources -> Study Dashboard -> Generate Source Study Pack -> observe server job completion -> expose reusable Study state`

This gate does not implement full background job history, server push, or direct generated-deck selection inside the flashcards controller. It verifies the PR-sized handoff from queued server job to visible completed pack state.

## Verified Behaviors

- Study Dashboard queues a server source study-pack job from selected Library note/media source items.
- After queueing, Study performs a bounded server status observation through `get_study_pack_job_status`.
- Completed job payloads with `study_pack` metadata update the dashboard with ready pack title, pack id, and deck id.
- Completed packs are visible in Recent Decks and enable Resume to open the Flashcards section for reuse.
- Queued/running jobs still keep the generation action recoverable; failed/cancelled jobs keep retry available with visible status.
- Local-mode source generation remains disabled with a server-mode explanation.

## Red/Green Evidence

Red test:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase3_source_study_generation.py -q
FAILED test_server_study_dashboard_observes_completed_source_pack_for_reuse
AssertionError: Timed out waiting for get_study_pack_job_status
```

Green test:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase3_source_study_generation.py -q
6 passed, 1 warning
```

Warning: existing `RequestsDependencyWarning` from the local virtualenv dependency set.

## Files Under Test

- `Tests/UI/test_product_maturity_phase3_source_study_generation.py`
- `tldw_chatbook/UI/Screens/study_screen.py`
- `tldw_chatbook/Widgets/Study/study_dashboard.py`
- `tldw_chatbook/UI/Screens/study_scope_models.py`

## Findings

No P0/P1 defects found in this gate.

Residual risks:

- Full server-side study-pack job history and long-running polling remain later workflow work.
- Direct generated deck selection inside the Flashcards controller remains a deeper Study interaction gate.
- Workspaces, Collections, and deeper Import/Export/Search/RAG study flows remain open under Phase 3.
