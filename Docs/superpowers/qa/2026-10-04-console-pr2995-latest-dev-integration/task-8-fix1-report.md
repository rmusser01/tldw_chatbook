# Task8 fix round1 — I1

Status: **DONE_WITH_CONCERNS**. Source frozen at `d3443b9e4297fa20897cf99e77a3ae2c8c562b10` from root checkpoint `72c67b80a870308e2425b5c2ad9437c3cfe01bd8`. Worktree and index are clean. Independent I1 review remains a root gate; the controller size failure remains blocking and requires the separately scoped structural repair.

## Change

Prepared creation now captures exact record/source/bridge/database/runtime identities under the existing plain registry lock, reads run/parent rows outside it, and rechecks exact payload/record identity plus current source, actor, incarnation, conversation, workspace, cancel, Close, revocation and current-primary fences under that lock. Approval and grant insertion stay inside the final critical section. The private observation is local to each phase. Fast remembered approval, post-wait approval, execution entry, final save and native source checks read anew; no authority boolean, cache, new lock, getter replacement or persistence under the registry lock was added.

The copied fields preserve absent keys, including the original fail-closed behavior for a missing status. Independent database writers can still change a status after observation. Both immutable 40b9 and corrected code demonstrate that existing observation boundary, and both refuse a terminal source on final pre-save revalidation without creating a row. This follows the committed ADR219 clarification; it does not claim a latest-database-at-decision transaction.

Only `console_chat_controller.py` and `test_console_chat_create_integration.py` changed. Four existing controller methods changed; four private helpers and one observation dataclass were added. Every preexisting integration-test function remains AST-identical. Fork, saved-draft, native acceptance and runtime/storage writer policy are unchanged.

## Verification

| Receipt | Result |
| --- | --- |
| `task-8-fix1-final-validation` | 24 passed: unlocked wrapper/fast/post-wait reads, real parked connection with owner-thread Close/revocation, missing-status refusal, observation race and final save refusal. |
| `task-8-fix1-final-callers` | 56 passed: all post-copy runtime invalidations plus 24 existing approval, grant, child cancellation, saved-draft and native cutoff neighbors. |
| `task-8-fix1-final-immutable` | Expected 5 failures and 4 passes: four blocked-read controls and the inherited cap fail on immutable 40b9; database-only observation and final-save controls pass for primary and child. |
| `task-8-fix1-final-cap` | One cap failure, one slack-row pass. No cap edit or waiver. |
| Final fatal Ruff, touched formatter ratchet, source whitespace | Passed. Formatting has full-module AST equivalence proofs. |
| Final diagnostics and worker reconstruction | Passed on committed source; unchanged diagnostic pins, 324 DOM lookups in 151 owners and 68 waits from 26 owners. |

The two final behavior receipts total **80 passed, zero failures/errors/skips** on exact committed source. Full commands and source-before/after hashes are in the JSON receipts. No full suite, dependency installation, skip/xfail, suppression, guard increase, push, PR message or merge occurred.

### Retained REDs and setup corrections

All intermediate failures remain in the separate safe manifest. Initial Close setup called an owner-only operation from another thread and produced `QueueThreadViolation`; the corrected control runs actual Close on its owner thread with a bounded release watchdog. The first GREEN used the raw validation helper as if it owned token cleanup; the final control uses real confirmation, parks its second read after attribution, and verifies actual denial/token retirement. Immutable 40b9 fails all four of these final controls at the intended registry-lock barrier, with bounded worker termination. Neither setup diagnostic is relabeled as the lock RED.

Self-review also caught `.get()` filling a missing copied status with `None`, weakening the original primary refusal. Its RED remains `task-8-fix1-red-missing-field`; the copy now retains missing keys and final controls pass. Earlier cap and intermediate verification receipts are retained verbatim.

## Remaining cap gate

The unchanged controller limit is **29,367 lines**. Immutable 40b9 is **32,910** (+3,543); fix1 is **33,027** (+3,660), a **+117** fix delta. Exact immutable/current failures are retained. Root directed a clean I1 freeze with this concern so a separate structural repair cannot obscure the concurrency change. No extraction or threshold change was attempted here.

## Carried qualification and warning disclosure

The prior **58 startup/navigation cases passed with three warnings**; they were not replayed in fix1. All selected startup test/guard/owner files are byte-identical except the controller's scoped validation bodies. Controller import ASTs are identical, no module was added, and the preimport guard explicitly excludes the prewarmed Chat/controller closure from its added-LOC count. The map states these limits; the 58-case receipt never qualified the newly inspected controller size row.

Retained budget warnings:

- Boot imports: **681/686**, headroom 5; snapshot drift +25/-13.
- UI-ready modules: **1033/1033**, headroom 0; drift +5/-2.
- Preimport: **557/557** modules, headroom 0; **415,298/425,347** LOC and Library **127,527/135,111** LOC.

Two Task8 child-process warnings omitted from the original report are explicitly carried here:

- Pending-projection group: **+265 FDs**, 14 → 279, limit 200; `carried-907b89d38dec3bf3-pytest.log`.
- Final surviving-child group: **+233 FDs**, 14 → 247, limit 200; `carried-4ecf6a78f65b90b5-pytest.log`.

These passing child tests emitted warnings. Attribution remains unresolved; the logs do not identify a responsible resource or establish a newly introduced/fixed production leak. No warning was suppressed or threshold raised. The earlier Task7 +223 warning remains historical evidence in the original Task8 report.

## Preservation and handoff

`task-8-fix1-freeze-map.json` contains **159** source/test/helper before/after maps, complete affected function/caller maps, identical production import ASTs, unchanged runtime/storage dependencies, all **34** Task7 function ASTs, and exact QA tree proofs: **11,572** required historical blobs, **11,750** checkpoint blobs, **42** incoming blobs and all **11,792** QA blobs present at 40b9. No mismatch was found. Existing census and CSS source/outputs are unchanged. The two-file corrective diff is `task-8-fix1-diff.log`.

Early immutable diagnostic receipts did not independently hash their imported integration-test file. The final immutable receipt repeats the final controls and records the complete inherited selected source/test/helper set before/after, plus immutable controller hash; earlier receipts are retained as diagnostics, not sole provenance.

`task-8-fix1-safe-evidence-manifest.json` lists **71** verified JSON/log/XML copies with absolute paths, SHA256 and byte counts. It includes the separate maps, every fix1 failure and the three carried warning logs. Raw profile/config/database/cache/basetemp directories are excluded. Original Task8 report, review and evidence remain unchanged. Root owns scoped re-review, cap repair, external gates and publication.
