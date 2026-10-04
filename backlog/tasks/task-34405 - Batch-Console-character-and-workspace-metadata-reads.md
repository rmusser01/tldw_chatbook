---
id: TASK-34405
title: Batch Console character and workspace metadata reads
status: In Progress
created_date: 2026-10-04 19:53
assignee:
- '@codex'
updated_date: 2026-10-04 21:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair measured root causes from TASK-34402 under the user-authorized combined performance fix pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Character scope refresh avoids repeated owned connection preparation within one finite snapshot
- [x] #2 Authority revision ambient-scope changes and cancellation retain all freshness and retirement guarantees
- [x] #3 Workspace availability refresh removes redundant reads and publishes only the current generation
- [ ] #4 A finite qualified database callback shares one complete counted repository interval without dropping SQL, extending native connection ownership, admitting fresh work during pause, or granting authority to other repositories or unqualified owners
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-10-04-console-performance-fixes.md. Reproduce RED, implement minimal fixes, verify targeted native and authority regressions.
1. Preserve the completed metadata batching and all ambient, revision and publication fences.
2. Add a real SQLite admission census comparing separate reads with one finite callback and retain actual SQL and native ownership evidence.
3. Enclose the exact file-backed run_owned_db_call callback in the existing repository counted operation, inside operation_owned_connection so its interval exits before new-handle retirement.
4. Verify borrowed transactions, cancellation and pause drain, already-counted continuation versus fresh same/other-owner refusal, independent cross-repository admission and custom/subclass/memory controls.
5. Run finite-retirement, metadata and local participant regressions only; record scoped lint and native evidence for the combined pass.
ADR required: yes. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: explicitly define the complete finite-callback counted interval using existing repository boundaries; same-owner nested reads retain their operation while other owners require ordinary independent admission. No authority cache or retirement bypass.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Character scope captures now keep both authority/revision pairs on one finite operation-owned worker connection. The midpoint ambient check still runs on the captured UI event loop through a bounded rendezvous; expiration cancels queued checks, a closed loop fails safely, and cancelled awaiters leave native retirement with the executing callback. Initial/final database, character, label and conversation checks, metadata equality retries and generation publication fences remain intact.

Availability capture reads the standard LocalWorkspaceRegistryService runtime set once per workspace and applies the existing disk-status recomputation within the same finite operation. Default stale-binding cleanup and non-folder runtime bindings remain represented. Custom/subclass adapters retain their own folder-lister and merge contract.

Files: tldw_chatbook/UI/Console_Modules/character_context.py; tldw_chatbook/UI/Console_Modules/workspace.py; Tests/Backup_Recovery/test_console_metadata_batching.py.

RED: real Windows/file-backed SQLite checks showed 2 native metadata handles instead of 1 and 4 binding SELECTs for Default plus named workspace instead of 2 (2 failed, 6 passed). GREEN: final focused native run 12 passed in 33.23s, verifying one retired metadata handle, unchanged paired SQL reads, two binding reads, real authority/revision and ambient switches, UI-only accessors, cancellation, expired/closed-loop retirement and custom folder status. POSIX helper preparation is observed by a call-through census when that native route is exercised; Linux/macOS verification belongs to the combined matrix.

Existing targeted character/async ownership/workspace/native-retirement run with the repository-supported bootstrap_profile marker: 207 passed, 3 failed in 510.42s. Removed-browser state-identity and constructor-docstring failures reproduce with isolated HEAD modules and printed import provenance. The grouped fresh-screen active-resume test failed in Textual Timer teardown with active_app LookupError; isolated HEAD and current reruns both pass, so this is recorded as an unreproduced teardown limitation rather than proven baseline. No full sweep was run.

Scoped Ruff checks and new-test/workspace formatting pass; the existing workspace E721 is identical in HEAD and excluded for this scoped lint run. Character formatter differences are in unchanged HEAD code. git diff --check passes. Self-review complete; status remains In Progress for combined native evidence and review.

ADR required: yes for the complete finite callback counted interval. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md (TASK-34405 amendment written before the boundary change). The earlier UI batching retains ADR-028 folder semantics and ADR-126 native ownership; the added interval explicitly documents same-receiver nesting, independent other-receiver admission and retirement ordering.

Harness qualification: the grouped mounted run remained alive after printing its 207/3 summary and required cleanup of the verified private pytest process. Its results are a printed-summary receipt rather than a clean process-exit receipt; the focused native 12-case run exited normally. Isolated HEAD/current active-resume reruns both exited successfully but emitted post-capture logging from background config seeding.
Further native investigation found that shared SQLite handles still reacquired ordinary participant admission separately around each database method. run_owned_db_call now retains the existing _core_operation around its complete exact file-backed CharactersRAGDB, AgentRunsDB or WorkspaceDB callback, nested inside operation_owned_connection so the counted interval exits before physical close. Every SQL statement still executes and each native allocation keeps its separate descendant lease. Existing borrowed handles/transactions, awaiter cancellation and maintenance drain retain their ownership semantics.

The actual workspace availability async worker now captures the registry service and its database, routes qualified WorkspaceDB reads through that same finite helper, and rechecks service/database identities plus the existing request generation before publication. A captured-registry read helper avoids a redundant inner owned close; direct/custom/subclass routes retain their original owned wrapper and lister contracts. Database redirection stops remaining workspace reads and retries with the current owner. This closes a real cross-await stale-registry publication and replacement-worker-handle leak found by the new controls.

RED: three real independent reads required 3 ordinary participant acquisitions instead of 1 in notes/runs/workspaces; availability Default+named required 2 instead of 1. Availability owner-change/cancellation controls reproduced stale named=True after registry replacement, an unretired replacement DB worker lease after database redirection, and missing complete callback counting (4 failed, 1 generation control passed). GREEN: final bounded interval24 cases comprises 23 passing in the combined 75-case run (63.58s, normal exit 0) plus the reviewer-requested actual VACUUM compatibility case (1 passed in 4.00s, normal exit 0). The combined run also includes existing finite DB retirement, metadata12, backup worker retirement/borrowed transactions/raw-closed caches and three lightweight existing availability/custom controls. Separate local participant controls: 16 passed, 44 deselected in 7.35s, exit 0. Successful VACUUM through run_owned_db_call shrinks the real file, records durable completion, preserves shared fork owners/lineage, reopens quick_check=ok and retires all worker/native/core resources.

The native census proves 3 method admissions -> 1 complete finite callback plus a distinct native descendant lease, with all 3 SQL reads retained. Availability retains 2 binding SELECTs for Default+named and exactly one physical close after the core interval exits. Memory and real uninstalled subclasses keep their standalone caller-owned handles; custom callback arguments/results and thread route remain unchanged. Existing security nuance: an uninstalled receiver nested inside an installed counted operation remains refused by the existing participant fence; the wider complete callback interval preserves this fence rather than suppressing discovery.

Additional files: tldw_chatbook/DB/base_db.py; Tests/Backup_Recovery/test_finite_db_counted_interval.py; ADR-126 TASK-34405 amendment. Final production SHA256: base_db B82730279CE481DE64F9AA22E828C522F6C4B08C0CDD1FC73235A94BC1C25F6F; workspace 502D963AEF4F2087244FD9A103B845FD132950A52C6076F25747E469482FAC8C; character_context 9E12616CE73033B1F57F412383BADA75E1D3FE92B1B5191A9FEB5BAA2987B064. Scoped Ruff/new-test and workspace formatting/diff checks pass; base_db seven existing exact-type E721 guards were verified against HEAD and excluded for scoped lint. AC4 and status remain pending the parent's combined native qualification/review; no full suite, commit or push.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
