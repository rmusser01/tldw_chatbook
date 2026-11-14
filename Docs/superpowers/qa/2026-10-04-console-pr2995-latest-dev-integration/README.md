# PR #2995 latest-dev integration

This follow-up preserves exact evidence for the user-authorized Console chat creation PR. Earlier QA in [the initial integration](../2026-10-03-console-chat-starts-dev-integration/README.md) and [review repairs](../2026-10-03-console-pr2995-review-and-merge/README.md) retains its recorded revisions and bytes. Counts below overlap and must not be summed.

## Final source and gates

- Reviewed final source: `9445ed93ed430ba0e3660ad2008621f6a809bd0e`.
- Integrated dev: `8f83422dde2a5b95da648882b8f7e09da5b41f08`; exact ancestry and later freshness evidence: `task-13-root-handoff-verification.json`.
- Publication/merge status at archival checkpoint: `Reviewed source ready for publication; current-head Qodo/CI and normal merge pending`.

This archive records source qualification. Current-head Qodo completion, required CI, PerfGuard and latest-dev ancestry remain external merge gates until separately confirmed.

## Frozen phases

| Phase | Evidence and boundary |
| --- | --- |
| Task6 runtime, fork and scanner | Source ownership/fork barriers and scanner escape repair; independent scoped reviews and exact source maps carry earlier controls. |
| Task6 schema/encryption | Preserve shipped notes75→76, add native76→77 receipts, qualify exact historical catalog variants. Twenty staged plus22 runtime cases were requalified with nine contemporaneous test/runtime/SQL hashes to close the review provenance gap. |
| Task6 model configuration | Pinned df2 source9ed370f566:74 contracts and five real UI journeys pass. Its startup cap failure and original fixture/formatter failures remain preserved. Task7 closes the actual Console cap defect; this phase alone did not. |
| Task7 provider adoption and ownership | Frozen19f2904496c0:40 adoption,20 hooks,3 DI/resume and3 cap/census controls pass within exact source/method/test maps. Both typed consumers use the canonical clean settings rebase. Adoption resides in Session, hooks adapter in existing wiring; Console25188/759 passes unchanged25218/759 caps. Independent review is Compliant/Approved. |
| Task8 latest-dev integration and I1 | Frozen d3443b9e4297:80 targeted controls pass with Compliant/Approved scoped review. Fresh storage observations stay outside the registry lock; exact source Close/grant/acceptance fences remain atomic. |
| Task10 store size and pure fork projections | Frozen1d55d8f89f:280 affected fork cases/2 store guards pass, actual22338/22344 store and1584 fork owner, combined source+132 lines. Six retained method patch routes and all mutable authority stay in the store. Independent review Compliant/Approved; zero introduced unrestricted findings. |
| Task11 latest dev and inherited admission fixtures | Frozen084b atf800:65 distinct cases pass; eleven canonical prologues/ten positive admission witnesses preserve original statements/assertions by exact reversal. Newest source a5bce325 onto1b12 (runtime9878) adds59 selected modal/fork/video cases, exact29851 source union and12337 historical QA. Original126 artifact hashes and extension21safe/2top hashes stay exact. |
| Task9 explicit decision/preflight/UI ownership | Frozen `3025df959c`; independent Compliant/Approved. 249 pins and exact body/182 Session maps verified. Chronological 560 distinct named cases have passing latest receipts; original failures and interruptions remain. Controller 29327/29367, host 6479/6479, compaction 4185/4185, screen 25192/25218 and 759 methods. Two bounded-await corrections and one admitted legacy summary correction change behavior. |
| Task12 latest dev durable delete, MCP and activation | Frozen `bda1dbf34b`: 83 selected cases and two derived checks pass; independent Compliant/Approved review. Root and reviewer verify 502 pins, 2,731 method rows, 29,861 source paths and 7,689 qualified source hashes. The upstream QA append remains additive. |
| Task13 latest dev send refusal and trace categories | Frozen `9445ed93ed`: 57 cases pass (53 behavior plus four ratchets); independent Compliant/Approved. Exact 1,699 declaration hashes, 8,123 source/test hashes, 518 prior evidence pins and 12,337 QA blobs verified. Four shared feature methods restore exactly after reversing incoming hunks. Controller row lowers to 29,301; other caps stay unchanged. FD growth 369 remains a warning with unestablished cause. |

The exact approved prepared draft is retained when source Close races successful persistence before native acceptance. Retired sources cannot place or launch it. After AgentRuns acceptance, source Close cannot undo target work, receipts or charges. The legacy fork route keeps upstream orphan cleanup. Approval finalization uses the existing non-reentrant lock without storage reads in its critical section. Task8 I1 captures exact record/runtime identities, reads and copies row fields outside that lock, then atomically rechecks current runtime/record/Close/revocation predicates. Each approval/save phase obtains new observations; this does not serialize independent database writers or promise a latest-row transaction. ADR219 records this compatibility before implementation.

Task8 I1 at d3443b9e4297 qualifies80 targeted controls with independent Compliant/Approved review. The root R1 addendum retains three explicit blocked-lock immutable failures and one fourth unexposed subagent-revoke error-list failure without claiming its exact cause. All original receipts are preserved. Task10 closes the store gate; the final Task9 controller/screen/owner measurements and independent verdict are recorded above. Historical failures retain their original status.

## Evidence and preservation

Task reports, immutable review packages, explicit source/test hashes, commands, original failures, pass receipts and independent verdicts are copied exactly. Named safe evidence manifests include only hash-verified child JSON/log/XML files and explicitly pinned verifier text; private profiles, configuration, databases, caches, bundles and unrelated plan artifacts are excluded. The publication manifest and audit describe this bounded selection.

`task-13-safe-evidence/final-source-union.json, historical-qa-carry.json and task-9-root-publication-preservation.json` maps the final upstream/feature source union and 12,336 unchanged historical QA blobs. One mapped document, `Docs/QA/task-31245/switcher-reuse-and-mode-follow-up.md`, retains its entire original 7,982-byte prefix and the exact upstream 4,417-byte/70-line append. Its original Git blob and frozen safe row remain recorded; all immutable phase receipts are preserved. The publication ledger snapshot retains controller rulings and outstanding conditions. No full repository sweep, dependency install, check bypass, new skip/XFAIL, warning suppression or feature budget increase was used.

## Limits

- Task8 receipts retain descriptor growth265 (14→279) and233 (14→247), limit200, with no attributed cause or warning suppression. Task7 hooks tests retain descriptor growth223 (14→237, limit200). Earlier teardown descriptor warnings and the historical external readiness timeout also remain disclosed. These records do not locate or repair a production leak; the timeout was not locally reproduced.
- Task9 retains descriptor growth257 (14→271, limit200) in its passed pending-projection journey, without an attributed production leak or suppression.
- Original failed runs remain failed evidence even when filenames contain GREEN. The reports distinguish setup attribution, actual behavior failures and later corrected controls.
- The24 historical model-integration formatter findings,97 inherited Task11 hunks and four newest-dev formatter edits keep their source-specific attribution. Final scoped formatting evidence is `task-13-report.md and task-13-safe-evidence/formatter-inheritance.json`. Unrestricted baseline lint exits retain their recorded status; passing fatal checks do not relabel them.
- Cold connection/WAL opening and FULL fsync during ledger admission remain synchronous. The warm writer-lock control qualifies immediate contention refusal at the unchanged cutoff.
- The older skill-await test isolates hooks review; separately qualified hook controls cover admission. Combined skill-await plus hook refusal remains a coverage limit.
- The root loading run at `a6887fdb89` passed 35 cases in 60.90s with 7,689 identical before/after source hashes. Measurements: app import 681/686 modules; UI ready 1,033/1,033; preimport 557/557 modules, 415,310/425,347 LOC and 127,527/135,111 LOC for the library route; CSS 607,326/608,090 bytes. Three parent headroom warnings and one private CSS child warning remain. Task13 loading carry is recorded in task-13-safe-evidence/loading-source-carry.json and loading-authority-detail.json; unchanged eager module identities, constructors, guards, Session authority and startup sources carry, while fresh mounted cases qualify changed presentation. This archive does not claim a fresh combined startup run after that source changed.

## Decisions and current scope

[Complete progress ledger](task-9-complete-ledger-at-publication.md) and [Rulings I made](task-9-rulings-i-made-at-publication.md) preserve all 61 ordered controller rulings and their costs if wrong. The integration task remains In Progress until actual external review/checks and merge complete.

Task13 preserves upstream documented residuals concerning post-mount trace maintenance, speculative voice media admission, Temporary recovery duplication and the Chats-list blocked marker. Its report states their precise limits. It introduces no assertion or policy exemption for them.
