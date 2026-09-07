---
id: TASK-15103
title: Reconcile 19-owner latest-dev diagnostic inventory drift
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-11 04:38'
updated_date: '2026-08-13 19:46'
labels:
  - testing
  - baseline
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Latest-dev stop-gate revalidation on exact `origin/dev` `82b595049d97836482c118cfeb4d31df537a86a1` expands the generated-versus-stored baseline to exactly 19 unrelated owner paths: `tldw_chatbook/Agents/agent_service.py`; `tldw_chatbook/Chat/console_agent_bridge.py`; `tldw_chatbook/Chat/console_chat_controller.py`; `tldw_chatbook/Chat/console_chat_store.py`; `tldw_chatbook/Chat/console_context_compaction.py`; `tldw_chatbook/Chat/console_provider_gateway.py`; `tldw_chatbook/MCP/client.py`; `tldw_chatbook/MCP/local_server_tools.py`; `tldw_chatbook/MCP/prompts.py`; `tldw_chatbook/MCP/server.py`; `tldw_chatbook/RAG_Search/fusion.py`; `tldw_chatbook/RAG_Search/simplified/rag_service.py`; `tldw_chatbook/RAG_Search/simplified/search_service.py`; `tldw_chatbook/UI/Console_Modules/session.py`; `tldw_chatbook/UI/Screens/chat_screen.py`; `tldw_chatbook/UI/Screens/library_screen.py`; `tldw_chatbook/app.py`; `tldw_chatbook/UI/Screens/settings_screen.py`; and, as the 19th path, `tldw_chatbook/Utils/text_selection_crash_guard.py`. The detached canonical `--write` Git-patch manifest diff is 53 additions/32 deletions with SHA-256 `adee369a60248da32fbc77c36b703618c73c61f5d5ef63d95460ada758f15a0f`; stored/generated totals are owner files 485/488, TASK-492 calls 1,144/1,180, TASK-494 calls 6,962/6,990, and persistent-sink files 6/6 with unchanged topology. Relative to the approved 18-owner population, every existing owner remains identical except `library_screen.py`, which moves from 84 calls/digest `c14a8222d35aec3a6e34` to 86/`ae0fac2e87bf1a6ee81c`, and the new Utils owner contributes one call/digest `f90a373ef5fcc81a8c1c`. ADR-029 inspection classifies the fixed Library trash-restore warning with only `type(exc).__name__` as reviewed-safe, the Library trash-load warning with exception capture as metadata repair, and the Utils warning with unbounded widget `repr` plus event coordinates as metadata repair. Review and reconcile this exact current-dev incident without using the refresh or any preliminary classification to bless unrelated drift. This record moved from the later-claimed TASK-14914 to TASK-15103 during the TASK-3796 PR rebase because exact add-commit provenance established that dev's visual-compaction task claimed TASK-14914 first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unsafe private values or exception details in the recorded delta are repaired without changing unrelated production behavior
- [x] #2 The persistent-diagnostic inventory is regenerated with only reviewed owner changes; the recorded incident preserves its six-file sink topology and any final-rebase topology delta is explicitly pinned and reviewed
- [x] #3 The focused architecture checker and regression coverage pass without constructing a test application
- [x] #4 Every generated-versus-stored delta on the final integration base is reviewed under ADR-029, including the recorded 19-owner incident and later rebase drift
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed executable plan:
`Docs/superpowers/plans/2026-08-11-task-15103-diagnostic-inventory-reconciliation.md`

1. Freeze the exact 19-owner incident in a schema-validated multiset ledger
   with per-group provenance and disposition evidence.
2. Replace the shallow TASK-15103 review path with a ledger-driven guard that
   reuses and hardens the existing alias-aware diagnostic extractor and
   mutation-tests all supported message/capture forms. Task 2 explicitly owns
   the proven alias/scoping gap for aliases introduced or mutated in `try`,
   `for`, `while`, `with`, and `match`, including conservative control-flow
   joins, shadowing, reassignment, and mutation coverage.
3. Add direct real-production-function privacy sentinels and repair unsafe
   Agents, Chat, MCP, RAG, UI, and application diagnostics in reviewed batches.
4. Regenerate the production diagnostic manifest only after all source and
   ledger gates pass, then prove the boundary rejects unknown data, forged
   summaries, extra owners, classification changes, and sink changes.
5. Rebase onto the latest `dev`, compare the complete call population for all
   19 owners, rerun only touched-function tests/static gates, complete
   independent review, and close the task.
6. For the final PR integration rebase, inspect every new generated-versus-
   stored owner delta without blanket regeneration, repair any unsafe call
   shape, address valid review feedback, then regenerate once and rerun the
   focused architecture/privacy gates before returning the task to Done.

ADR required: no

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: this task enforces the existing ADR-029 privacy boundary without
changing persistent-sink ownership, storage, or the metadata policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resumed after the Task-1 evidence phase (ledger schema + forgery matrix,
commits `93341871a..3b663504a`) with the execution order inverted: production
repairs first, one manifest regeneration at the end, instead of re-freezing
the boundary against every dev advance.

**Repairs (Tasks 3-6 equivalent, four commits):** all 43 metadata-repair
groups across 12 files were repaired to their frozen ledger contracts —
`logger.<method>("<fixed_event>")` plus only ledger-permitted expressions; no
exception capture, no interpolated ids, no `.bind()` fields. The 12
justified-deletion groups were verified already-resolved in dev history, and
G091's proposed intermediate is consumed by reviewed-safe G062 (chain nets
out; no edit). Every repaired call was digest-verified against
`proposed_surviving` with the gate's own extractor. Six test files that had
codified the leaky shapes (structured compaction/decision extras, value-named
RAG warnings, crash-guard widget repr) were updated to the event-only
contracts. In `session.py`/`library_screen.py` the repaired calls use an
unbound loguru alias because the frozen contracts carry no fields — the
file-level `module=` bind is folded into digests by the extractor, and
nothing in Logging_Config consumes that extra.

**Close-out (Task 7 equivalent):** ledger flipped to `reviewed` with
mechanically derived `final_base` + per-owner `reviewed_final` pairs;
manifest regenerated once — exactly the 19 recorded owner rows changed,
sinks byte-equal, totals 485→488 / 1,144→1,180 / 6,962→6,990 as recorded.
The canonical node is now status-aware (stored-vs-live delta is a
planned-lifecycle assertion), and a new
`test_task_15103_reviewed_final_state_is_ledger_exact` node fail-closes the
final state against the real inventory: manifest == live, `reviewed_final`
== live, and per-digest `live == max(0, recorded + ledger net)` — the clamp
is exact because drift-introduced atoms appear in groups only as removals.

**Deviations from the written plan:** the per-batch privacy-sentinel test
files and the five apply_patch manifest mutants were not built; final-state
enforcement is carried by the reviewed-state node + the existing forgery
matrix instead. Task 2's full alias/scoping extractor hardening remains open
(tracked in the plan) — the known control-flow alias gap is documented and
did not affect any of the 101 ledger groups.

**Defect found in the shipped Task-1 gate:** `_task_15103_complete_history`
read its stored baseline from the LIVE manifest, so the first legitimate
regeneration sent the stored-revision scan hunting for post-repair
populations that exist in no dev-reachable revision (and onto a
conflict-markered historical blob). It now reads the immutable
`recorded_base` tree, which matches the stale baseline on all 19 owner rows.

**Final integration rebase (2026-08-13):** rebased PR #1544 onto `origin/dev`
`1c4d25fc53224efee8bc698d6a5a514c72eed483`, reviewed all 61 resulting
owner-row deltas, and pinned their outcomes in
`Docs/security/task-15103-final-integration-review.json`: 28 metadata repairs,
27 safe/no-edit rows, and 6 removed owners. Raw config values,
survivor/tool/artifact identifiers, stack traces, exception capture/text,
message/profile/note ids, paths, and dynamic setting values were replaced with
fixed events plus only reviewed type/count metadata. The regenerated manifest
records 489 owners, 1,198 TASK-492 calls, and 6,940 TASK-494 calls. Its one
sink-topology delta is the reviewed removal of the redundant buffered Loguru
bridge already prohibited by TASK-15422; the final-integration ledger pins the
exact base/reviewed topology hashes and removed sink. The focused ledger test
also mutation-proves that a safe/no-edit row cannot disappear. Qodo's valid
feedback was addressed by adding the missing public-test docstring and
batching/deduplicating provenance Git checks; the event-only logging-context
suggestion remains intentionally rejected under ADR-029.
<!-- SECTION:NOTES:END -->
