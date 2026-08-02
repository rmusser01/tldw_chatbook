# Library Ingest P2 Batch (task-2015) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Executed inline by the planning session (full survey context in the session); each item is TDD'd and committed separately where meaningful.

**Goal:** Land the 12 confirmed P2 findings from the 2026-08-02 ingest UAT (task-2015) as one PR on `fix/library-ingest-uat-p2-2015` (base origin/dev @ `db493a89d`).

**Architecture:** All state-derivation changes go in `Library/library_ingest_state.py` (single source of truth for both queue-row surfaces); behavior changes in `UI/Screens/library_screen.py`; classification in `Local_Ingestion/`; copy in the canvas. No schema changes.

## Global Constraints

Same as the P1 plan (worktree-only, venv pytest, explicit staging, no stash, regenerate bundle never hand-edit, Co-Authored-By footer). Live verification + `/impeccable critique` re-run happen after 2016.

## Items → anchors (surveyed 2026-08-02, post-P1 dev)

1. **Debounced while-typing validation** — `handle_library_ingest_path_changed` (screen): cancel/restart a 0.8s `set_timer` that calls `_trigger_library_ingest_preflight(path)` when the path is unchanged and non-empty. Test: changed-handler sets timer; firing applies preflight; superseded timer doesn't.
2. **Above-fold completion signal** — `_handle_library_ingest_registry_changed`: track active count (queued+parsing+writing) via `registry.counts()`; on >0 → 0 transition, `app_instance.notify` "Ingest finished — N imported · M failed" using deltas vs the baseline captured on 0 → >0. Test: simulated transitions produce exactly one notify with correct counts.
3. **Empty-file failure permanent** — `_reject_empty_extraction` (`local_file_ingestion.py:921`) raise type gets classified permanent in `classify_parse_failure` (`ingest_parse_worker.py:72`), via a dedicated exception subclass. Test (runner): empty file → failed row with `permanent=True` / `can_retry=False`.
4. **Folder done rows carry media ids** — believed already fixed by task-2013 (dup case explained the UAT observation). Prove: extend directory runner test asserting every done job has `media_id`.
5. **Pluralization** — canvas unsupported-summary f-string: article agrees ("recorded as failures" plural).
6. **Unwrap nested failure copy** — `short_ingest_error` (`library_ingest_state.py:91`, shared by queue row + Home): collapse repeated `Failed to <verb> <type> file:` prefixes to one. Test: the verbatim triple-wrapped PDF string renders single-prefixed.
7. **Clear finished confirm** — two-press arm pattern on `#library-ingest-clear-finished` (screen handler :13082 + canvas button label from state): first press arms ("Press again to clear N finished…"), second executes; arming resets on registry change. Test: single press clears nothing; double press clears.
8. **110-col ellipsis** — `LibraryIngestCanvas CollapsibleTitle { text-overflow: ellipsis; }` (Textual 8.2.7 supports `text-overflow`) in `_agentic_terminal.tcss` + bundle regen. Verified live in the post-2016 pass.
9. **No estimate noise under errors** — state builder :836: when `errors` non-empty, suppress `estimate_line` + `type_breakdown_line`. Test: PreflightResult with errors → state has empty estimate/breakdown.
10. **Start disabled for guaranteed failure** — state :814: when preflight present, `total_files > 0`, and no supported type groups → `start_enabled=False` and `start_quiet_line` explains ("Nothing in this selection can be imported — N unsupported file(s)."). Test both the disable and the copy; a mixed selection stays enabled.
11. **Honest elapsed** — `_format_elapsed` base becomes the job's `submitted_at` (falls back to `started_at`; both absent → "" and the row omits the segment; sub-second → "<1s"). Guard restored jobs (submitted_at 0.0 → treat as unknown). Test: submitted→finished delta; 0.4s → "<1s"; unknown → no ` · ` tail.
12. **Per-stage progress** — parsing/writing rows already render `● parsing/writing · name` (state :543-574). Prove with a state test if none exists; tick with evidence.

Each item: red test → implement → green → targeted suites (`Tests/Library/test_library_ingest_state.py`, `test_library_ingest_runner.py`, `Tests/UI/test_library_shell.py -k ingest`) → commit. Final: full `Tests/Library` + shell file, task-2015 ACs ticked with notes, PR to dev.
