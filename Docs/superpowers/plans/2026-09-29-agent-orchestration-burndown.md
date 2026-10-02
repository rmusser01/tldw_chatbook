# Agent orchestration burn-down implementation plan

**Goal:** Close the remaining routing/progress defects, reconcile completed tickets, and deliver the requested optional orchestration capabilities under explicit reviewed contracts.

**Implementation base:** dev 64579cce2c8dc64053fb50c00eb4f59b56716b01.
**Integration base:** dev ab4df99959545e37d8d2048c1c1ce15fd914721d (PERF-06), following the qualified compaction and docs-only integrations below.
**Authority:** User requested all listed items on 2026-09-29. Existing task acceptance criteria govern each repair. Existing ADR-134/135/136/146/147/155 govern budgets, delivery, communication, routing and recovery.

## Global constraints

- Use the isolated codex/agent-orchestration-burndown worktree; preserve other checkouts and work.
- Preserve endpoint identity independently from execution spellings; explicit built-in child routing must still select the built-in destination.
- Preserve configured caps, cancellation, approval and physical-owner boundaries. No automatic replay of uncertain effects.
- Diagnostics and ordinary metadata contain no communication bodies or private URLs/paths.
- Follow ADR-150/161 UI tokens and existing rendering patterns; targeted verification only.
- Reproduce defects before minimal fixes; review changes independently before closing tasks.

## Repair sequence

- [x] TASK-32929: thread raw selection identity into primary run dispatch and keep explicit child target resolution independent. ADR required: no; direct repair under ADR-146/147, with clarification of the existing identity contract.
- [x] TASK-33001.9: reuse GENERATION_FIELD_REQUEST_KEYS and verify top_p through real provider projection. ADR required: no; implement the existing shared generation-field contract.
- [x] TASK-32639 / TASK-32517: keep count/navigation synchronization active when the button is absent; mounted regressions for removal/remount. ADR required: no; lifecycle bug fix.
- [x] TASK-32499: reconcile admission-refusal counting; surface routing level; move parameter parsing to existing Chat/sampling_params.py. ADR required: no; preserve ADR-147 behavior and consolidate a shared helper.
- [x] TASK-32497: project saved resolved provider/model for live/historical rows with honest legacy fallback and rendered regression. ADR required: no; direct ADR-147 follow-up.
- [x] TASK-22061 / TASK-32477 / TASK-32520: compare existing implementation/evidence to current criteria, verify affected behavior, repair remaining gaps and reconcile task status.

## Requested extensions

- [x] TASK-32508: specify and implement explicit pre-tool provider fallback chains. ADR required: yes; amend ADR-147 or create a focused canonical successor before implementation; keep budget/continuation/snapshot semantics explicit.
- [x] Direct sibling addressing, durable progress inboxes and progress-triggered wakes: inspect existing communication and delivery owners; create atomic Backlog tasks and ADR/spec before implementation. Reuse existing queues, SQLite ownership and wake budgets. Restart persistence preserves messages, never live execution authority.

## Completion

- [x] Targeted combined verification, static/derived checks and independent code review.
- [x] Update the workstream ledger with exact dispositions, acceptance evidence and remaining limits.

## Extension decisions and execution order

ADR required: yes
ADR paths: backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md; backlog/decisions/200-preset-pre-tool-fallback-targets.md
Reason: explicit new communication authority, persistent state and fallback policy.

1. TASK-32508: frozen preset fallback targets, schema migration and existing-runtime integration.
2. TASK-33430: scoped peers using existing steering and private metadata.
3. TASK-33431: durable inbox after fallback schema settles.
4. TASK-33432: progress wake source after durable enqueue is authoritative.

All four are in the authorized burn-down; run targeted checks and independent review for each.

## September 29 review checkpoint

All 13 tasks are Done with checked acceptance criteria and implementation notes. Durable progress passes 247 affected checks, 82 final native/modal/hydration checks and 74 final report/tool/queue checks; these overlapping selections are not summed. Independent final review approves all four worker cache corrections with 13 checks and actual participant drain. Routing/UI and schema/ledger selections pass 116 and 137 checks. Changed-code static verification passes; existing size-ratchet and source debt remain disclosed. The clean latest-dev rebase passes 217 targeted provider/search/routing/sampling/startup checks in 87.14s; changed-code checks remain clean. Delivery is through a draft PR against dev. No merge is claimed.

## October 1 CI and integration checkpoint

Rebased PR #2918 onto dev `31d4f9b76492120706ba8e3ad7d355f1aa4e0273`. Reopened TASK-33430/33431/33432 for their existing acceptance criteria and CI qualification. ADR required: no; direct repairs and verification under ADR-173/199.

- [x] Preserve exact runtime-tool inventory, including both scoped peer tools.
- [x] Use the existing canonical UTC millisecond timestamp writer for durable progress.
- [x] Pin real populated FIFO and progress-claim cleanup query plans, register both indexes and allowlist the new Chat table.
- [x] Audit changed production diagnostics before refreshing their inventory.
- [x] Repair the upstream hook-refusal early return through the existing exact wake authorization and preacceptance refund path; qualify real plain/agent retry and nonreplay.
- [x] Independently review overlapping runtime/routing/recovery boundaries and run final targeted checks.

Affected messaging/schema checks pass 145 cases; fallback/sampling/saved-close/hydration/mounted progress/token/guide checks pass 147 cases. The exact CI correction selection passes three cases. Hook repair passes six authority/real-gateway cases and three existing ledger guards; independent review passes both real provider paths and three ledger guards. The upstream MCP click correction passes all four focused cases. These selections overlap and are not summed. Final static qualification covers 75 changed Python files and ten new files with zero added-line/new-file findings. Delivery remains PR #2918 against dev; no merge is claimed.

## October 1 Qodo and boot CSS follow-up

- [x] Preserve the existing boot CSS ratchet by consolidating the two preset editor selectors into one scoped class and qualifying actual painted fields at standard/narrow widths.
- [x] Complete public peer documentation and provider-error annotations.
- [x] Add strict Pydantic peer argument validation through the existing text validator; preserve refusal codes, quotas, privacy and capability custody.
- [x] Replace global-close mutable membership traversal with existing immutable published membership after the global authority fence; qualify controlled removal and actual mounted disposal.
- [x] Independently re-review all four Qodo findings and close the reopened tasks after fresh targeted/static/diagnostic checks.

ADR required: no; direct qualification and mechanical repairs under ADR-097/150/173/199/200. All 13 tasks are Done. Fresh evidence:17 CSS/authoring/token/bundle cases, 88 peer/inventory/fallback cases, 2 startup guards, 44 queue cases and 4 mounted lifecycle cases. Independent final review approves 22 focused cases; selections overlap and are not summed. Final static scan covers 76 changed Python files and ten new files with zero added-line/new-file findings; diagnostics match the audited inventory. Publish the follow-up to ready PR #2918 and require fresh checks on that head. No merge is claimed.


## Final latest-dev qualification

Rebase base: dev `84247cb8435fcf59b6d8e2d97c6b2f0913934dd0` (PR #2948).
ADR required: no. This is integration of the existing readiness repair with ADR-199/200 behavior; no boundary or authority change is planned.

- [x] Rebase and independently review the incoming active-run readiness overlap.
- [x] Verify the 78-file changed-code static boundary and owned test-range formatting against the final base.
- [x] Investigate the combined failures and qualify the corrected preparation/publication oracles with fresh affected checks and precise negative controls.
- [x] Prepare the verified rebase for exact-lease publication; confirm the PR head and fresh CI state immediately after the push.

Final evidence: 13 affected checks pass in 177.22s; independent review approves five corrected controls in 209.78s. Frozen terminal painting and never-first-chunk controls fail at their intended assertions. The stopped 138-pass/7-fail selection is retained as diagnostic evidence, not claimed green. Production behavior and limits remain unchanged; all 13 tasks are Done. Fresh GitHub checks remain part of PR delivery.


## October 1 compaction migration integration

Dev shipped Chat schema 74 for auxiliary failure reasons before the unmerged durable-progress migration. Preserve the shipped 73→74 method and SQL exactly; compose progress as 74→75 with unchanged DDL and exact current Chat/shared-Subscriptions recovery gates. ADR required: no new ADR; amend ADR-199 and preserve ADR-052. Current-only Chat recovery and frozen AgentRuns 18/21/22 migration policy stay unchanged.

Qualification: genuine 73/74 upgrades retain saved conversations and auxiliary failure reasons; injected failure after progress table creation rolls back table/index/version and allows a clean retry. Exact fresh/core/shared catalogs, both shared stamp gates, durable queues, atomic Save and mounted reopen/close pass. The first affected selection had 29 passed/2 new fixture failures; the final selection passes 31 in 82.20s after using the supported read-only owner and staging immutable recovery candidates through the existing backup API. No production recovery gate was weakened.

Independent verification passes 30 core/dictionary/AgentRuns/standalone-Subscriptions checks and five frozen AgentRuns history checks. Incoming compaction/refund/nonreplay qualification passes 10 cases in 82.63s. Mounted saved reopen/close passes two cases in 18.15s; startup/CSS/guide passes four in 51.29s, imports 679/686 and UI-ready 1032/1033. Selections overlap and are not summed. The audited diagnostic inventory remains verified.

The subsequent dev 922440b93e changes only canonical ADRs, including ADR-210's accepted Console migration plan. Its step 6 retains the current Model/Agent rail sections until their replacement; this patch does not implement that separate redesign. Rebase without changing runtime code, verify its source manifest, then publish PR #2918 for fresh remote checks. No full suite or merge is claimed.


## Requested merge

Rebase onto dev `ab4df99959`, qualify the incoming warm-config behavior with targeted checks and independent review, publish the exact reviewed head, wait for Qodo and fresh required CI, address any new finding and merge under normal branch protection. ADR required: no; integration of the accepted PERF-06 behavior without changes to orchestration contracts. Existing qualification limits remain. The fresh 19-case integration, three startup/CSS guards and five independent consumer/fallback cases pass; all preceding patch Python bytes remain unchanged.


## October 2 final merge base

Latest qualified dev: `113e435ab0bff12e89ae1c9f06ec154f32f776bf`, following queue/logging/maintenance/freeze base `ee1c1e7365`. ADR required: no new ADR; integrate the accepted ADR-126 admission-evidence amendment while preserving ADR-199/200 and all existing authority, migration and recovery boundaries.

Rebase, inspect overlapping runtime paths, verify source-byte preservation, run targeted consumer and independent checks, record failed setup/optional-dependency qualification limits, publish with the exact remote lease, then require fresh Qodo and all four CI jobs before head-pinned normal merge. The approved five-minute heartbeat stops after verified merge. All 13 implementation tasks remain Done; no full sweep is authorized.


## TASK-33647 — latency integration repair

Keep the existing two-admission maintenance budget by sharing the installed core repository operation across the read-only probe and immediate-write fallback. Pin a genuine cold file-backed callback and retain the existing full census, idle read-only, lease refusal, rollback, parking, worker lifetime, startup and exact recovery checks. ADR required: no new ADR. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: no schema, authority or runtime boundary changes; reuse an existing admitted operation. Implementation and independent review are complete; fresh-head CI/Qodo and protected merge remain.


## Final Delete/Undo merge base

Latest qualified dev: `eba4305d8389a2112c99ac19fa804e9abea394ba` (PR #2941). Preserve incoming exact tombstone restore and native context fencing; compare the 81-file source manifest, run targeted persistence/Save/close/storage/startup checks and independent mounted/nonreplay review, then publish TASK-33647 with the exact observed lease. These checks and review pass. ADR required: no new ADR. ADR path: backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md and backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: integrate accepted Delete/Undo behavior and reuse existing repository admission without changing orchestration, migration or authority contracts. Fresh-head Qodo/CI and protected merge remain; do not repeatedly rebase while checks run merely because dev moves.


## Required keep-alive and project-folder integration

All four jobs pass on published `e8474ca134ab73d66ab36d168bf0a11a78367c6b`; its exact-head Qodo report resolves all four findings with zero open threads. GitHub reports BEHIND after dev advances to `6958e8dfa99a66680b1aec09fed65df0f1a91955` (PR #2944 / TASK-33621.13). Live branch protection requires strict up-to-date checks and enforces administrators.

1. Preserve upstream first-send project controls, Inspector worker/modal handling, dead-pump retirement and worker-contract census during the required rebase; compare all 81 qualified patch Python bytes.
2. Qualify the actual first-send/Save/close/wake overlaps and startup/diagnostic/worker-contract guards with targeted runs and independent read-only review. Change production only for verified integration defects.
3. Record evidence, publish once with the exact `e8474ca134ab73d66ab36d168bf0a11a78367c6b` lease, then require fresh-head Qodo and all four jobs before head-pinned protected merge. Do not chase further dev movement while those jobs run.

ADR required: no new ADR. ADR path: backlog/decisions/069-console-project-instruction-local-state-and-preflight.md and backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md. Reason: preserving integration of an accepted upstream repair without a new persistence, authority, lifecycle or orchestration decision. Existing ADR-126 worker/storage and schema/recovery constraints remain intact; no full sweep is authorized.

Qualification complete on `51b46e279578c9220c1d1f34711815dab271ebfa`: ten targeted consumers, four unchanged storage/startup/CSS guards, four independent persistence/Inspector cases and six independent lifecycle cases pass. Both source reviews approve. Final 81-file/new-file static, worker-contract and diagnostic guards pass; all CSS and all preceding patch Python bytes except the upstream store remain unchanged. Import/UI counts are 680/686 and 1033/1033 without pin changes. Publish once; fresh-head gates and protected merge remain.


## Required hook-review conflict integration

Published `64d77044761dbfd4993cdad7fcd12cc987c14cc9` passes all four jobs and exact-head Qodo resolves all four findings with no open threads. Dev `30ca4552b3b881bee6c077db08c44233e5e97e6a` ships PR #2945 / TASK-33621.28, and GitHub reports CONFLICTING / DIRTY.

1. Rebase onto that observed dev head and preserve both sides of any actual conflict, including testing lessons, the worker-owned human hook-review continuation and our exact wake refund/nonreplay gates.
2. Compare the 81-file source manifest; qualify the new accepted-status/modal/cancellation boundaries and existing hook-refund/Save/close consumers with targeted checks and independent review. Preserve schema, recovery catalogs, physical worker custody and all ratchets.
3. Record evidence and publish once with the exact `64d77044761dbfd4993cdad7fcd12cc987c14cc9` lease, then require fresh-head Qodo and all four jobs before normal protected merge. No full suite or preemptive rebase while CI runs.

ADR required: no new ADR. ADR path: backlog/decisions/148-console-run-hooks.md and backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md. Reason: preserve accepted upstream human review behavior and existing orchestration authority; no new schema, permission or lifecycle contract is planned. Existing ADR-126 worker/storage boundaries remain.

Qualification complete on `932d80aa56f34735b2168df4c76b2267d16b6daa`: sole lesson conflict preserves both exact append blocks; only upstream ChatScreen changes among 81 qualified patch files. Twelve real consumers, four unchanged storage/startup/CSS guards, six independent controller cases and nine independent mounted modal/diagnostic cases pass. Both source reviews approve; final static/new-file, worker-contract and 634-owner diagnostic guards pass. Original module pins and authority remain unchanged. Harness pump responsiveness does not qualify or close upstream TASK-33621.28's real-app Ctrl+Q criterion. Publish once; fresh-head gates and protected merge remain.


## Required maintenance-probe integration

All four jobs on published 8cf1014cbf4b6bc62da9823ef1707a96164fad2a pass; exact-head Qodo is clear, with zero unresolved threads. Live strict up-to-date protection requires including actual dev 52620c3a08bfe6057aa58b9319efb4ec34a40213 (PR #2920).

1. Preserve the incoming 1 Hz native maintenance probe and its two existing interval oracles during the required rebase. Compare all 81 previously qualified Python files and CSS.
2. Run the actual unpatched-monitor interval/cancellation/same-task coordination cases and targeted orchestration maintenance/close/cold-admission consumers. Obtain independent read-only review, retain original gates and qualify startup/storage only proportionately.
3. Record results and limits in existing qualification records, publish once with the exact observed 8cf1014 lease, then require fresh-head Qodo and all four jobs before protected merge. No full sweep or preemptive rebase while jobs run.

ADR required: no new ADR.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md and backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md.
Reason: preserving the accepted upstream probe interval without changing native custody, pause authority, SQL admission, migration, recovery or orchestration contracts. Do not close upstream TASK-33560: its remaining guarded-read/cache/performance criteria remain unchecked.

Qualification complete on `d4b3311586568d9ae1d66839731e40938f504528`: independent review reopened existing TASK-33432 AC3 for a proven early maintenance refusal. The exact-authority guard and one progress/completion regression repair it using existing cleanup; two controlled RED cases, 11 repair/neighbor cases, seven maintenance caller/drain cases and an independent real refund/resume probe establish the result. The original source census is restored and passes; its earlier unexplained typing-helper failure remains disclosed. Static/new-file/owned-format/worker/diagnostic checks pass, original module pins remain. No new ADR; existing ADR-126/199. Fresh published-head gates and protected merge remain.


## Required expanded-hooks integration

Actual dev e92b01515f9547aab2cfb14cd4d93a7a07419775 (PR #2946) makes published43b2059 CONFLICTING/DIRTY; no Actions run exists on that head. This necessary update is not a rebase while checks are running. Reopen TASK-33431 AC4 and TASK-33432 AC3/4 before implementation.

1. Preserve upstream compaction73→74 and hook receipt74→75 bytes; move unmerged progress to75→76 and update exact fresh/upgrade/recovery/current stamp gates. Keep frozen AgentRuns18/21/22 and current23 unchanged.
2. Resolve actual overlaps while retaining the exact early unaccepted-wake refusal guard, existing manual/automatic priority, hook requirements, peer constraints, native close and physical worker/resource custody. Reuse existing owners; do not broaden into plugin programs or keybinding repairs.
3. Qualify real SQLite73/74/75 migration and rollback, restricted/current-only recovery, durable Save/reopen, exact refund/resume/accepted nonreplay, incoming Stop continuation/required barriers and scoped child/tool consumers. Use targeted checks and independent read-only schema/runtime review. Preserve failed evidence and all prior qualification limits.
4. Verify owned static/artifact/startup guards, record exact evidence, close affected criteria through CLI, and publish once with exact43b2059c6761eedaaf1722f7d505b0c017f66139 lease. Fresh Qodo and all four current-head jobs precede normal protected head-pinned merge.

ADR required: no new ADR. ADR path: backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md (version clarification amended before code); existing ADR-052/126/162/163/197 govern. Reason: compose already accepted storage, scheduler, permission and lifecycle boundaries without changing owners. No dependency installs, gates/ceilings/warmup/work changes, full suite, admin bypass or foreign checkout cleanup.


### Verified child admission overlap

Independent real FleetChat/SQLite probes on the composed temporary tree prove retained resume bypasses SubagentStart: the pass path starts a second child but emits only one event, and a now-denying required handler still permits the second row/provider. Reopen incoming TASK-32679 AC2 and TASK-33430 AC1/4 before repair. Reuse one existing initialization block for spawn and resume, including the prospective peer inventory, before physical admission. Preserve current narrowed tools, budgets, context, authority, frozen routing and slot refund. ADR required: no new ADR; existing ADR-163/162/199 apply.

The same reviewer proves actual saved/granted standalone SessionEnd is refused during host disposal, while exact session-close succeeds and all physical cleanup/tickets settle. Both disposed host and closed HookPermissions contribute; an event-name exception alone is insufficient. This upstream notification behavior is a separate permission-lifecycle limit, not a new SQL/custody failure or permission to reopen ordinary authority. Evidence: /private/tmp/pr2918-expanded-review-runtime.log and.xml (3 failed/1 passed,15.27s). No real-app Ctrl+Q or broader plugin qualification is claimed.


### UI-ready import overlap

Unchanged UI-ready census on root is RED1034/1033; exact incoming e92 archive is GREEN1033/1033. Audit-hook stacks identify the extra fleet_messages import in progress_pending_metadata before the conditional no-inbox return during controller construction/seed_progress_hints. Express that existing conditional return early and move its import after it, preserving saved preparation and exact authority; qualify the unchanged census and actual pending/refund/Save consumers. ADR required: no new ADR; existing ADR-097/199 apply. No pin, warmup, count or measured-work change.


### Final observed modal-quit base

Dev advanced to fccf70d3b0cd21b0d44a906ec3a68b0633887684 (PR2949 / TASK33622.10) before publication; published43 remains conflicted with no current Actions run. Preserve this required update without changing its quit/keybinding policy. Compare the qualified schema/child/runtime source, run only actual quit/Console cleanup overlap checks plus changed startup guards, and obtain immutable final schema/runtime review before exact43 lease publication. ADR required: no new ADR; preserve ADR-031 and the incoming task, existing ADR-126/163/199 apply. Do not close separate TASK33621.28 or broaden into keybinding repair.


### Mounted qualification isolation

The final two mounted consumers produce one saved-reopen pass and one raw_source_selection_changed refusal inside _build_test_app/load_settings before live-report behavior. The latter test constructs the real app after Tests/UI per-test profile selection but imports production config during collection; its stale raw participant correctly refuses. Reuse existing private_profile_test for this one owned mounted test, preserving all Save/selection/arrival/revocation assertions and the production recovery guard. ADR required: no; test-only setup isolation, existing ADR-126/199 unchanged. Retain /private/tmp/pr2918-expanded-final-mounted.log as NON-GREEN.


Qualification complete at3d14b4a: independent schema41 cases and runtime19 cases approve; actual v2 wake refusal/refund4 cases pass. Root10 repair,24 runtime, final4 originalstartup/storage,7 modal-quit and corrected actual mounted checks pass. All affected tasks are Done through CLI. Minimal child admission and no-inbox import corrections preserve existing owners; supplemental private-profile wrapper is test-only and independently approved. Detailed positive/non-green evidence and separateToDoTASK33648 are recorded in the final review. Doc-only publication cut must retain every reviewed production byte, then require fresh exact-head Qodo and all four CI jobs before protected merge.


## Required Resend merge-base integration

Published `003a7bc940458661654547d59e9adab514da6461` has all four required CI jobs passing and exact-head Qodo clear. GitHub reports BEHIND and live dev protection requires strict up-to-date checks. The actual fetched base is `bb865f5cfe` (PR #2956 / TASK-33661), including the earlier documentation-only PR #2952; no current-head job remains running. Preserve this necessary update once.

ADR required: no
ADR path: backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md; existing ADR-126/163/197 and ADR-031/097 apply.
Reason: integrate accepted Resend and custody-poll behavior without changing schema, permission, scheduler, storage or lifecycle owners. Preserve schema76, both current stamp gates, frozen AgentRuns history, exact maintenance-wake refund and physical worker custody.

- [ ] Rebase onto the observed dev commit, preserving all upstream source and complete append-only lesson tails; compare `/private/tmp/pr2918-pre-resend-manifest.json` to isolate actual overlap.
- [ ] Qualify Resend maintenance refusal, required-hook refusal, in-place retry/Continue, cancelled refused-echo recovery, original wake refund/accepted nonreplay and saved-close consumers using real runtime/store tests. Run the original startup/storage/CSS guards and focused mounted custody/Resend checks; do not change pins, ceilings or measured work.
- [ ] Obtain independent read-only reviews of the runtime/authority and mounted dispatch/polling overlaps on an immutable candidate. Keep incoming TASK-33662 real-relaunch recovery and earlier TASK-33648 SessionEnd suppression as separate disclosed limitations. Do not change other programs' task status.
- [ ] Verify static/artifact guards and preserved reviewed bytes, record exact green/non-green evidence, then publish once with the exact observed `003a7bc940458661654547d59e9adab514da6461` lease. Require fresh exact-head Qodo and all four jobs before normal protected head-pinned merge.
