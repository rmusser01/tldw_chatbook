---
id: TASK-34601
title: Cut Console Send latency by measured trace-fix-measure iterations
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-08 11:20'
updated_date: '2026-10-10 05:16'
labels:
  - performance
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console Send on native Windows takes ~6 s (warm) to ~7 s (cold) from the Send action to the provider adapter entry, excluding provider time, against the agreed targets of under 100 ms to first rendered feedback and under one second of application overhead (ADR-225). Earlier work fixed many correctness issues but never demonstrated a whole-Send latency gain. This task runs the loop the owner asked for: trace the current Send, fix the largest measured removable cause, re-measure the same scenario under comparable conditions, repeat. Each iteration ships separately with clean before/after evidence, so every gain is attributable to one change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The native Send probes run at the current source again (harness compatible with the received-intent preparation keyword)
- [ ] #2 Each shipped iteration has interleaved, observer-free native before/after receipts of the same scenario with exact revisions, and the gain is attributed to that one change
- [ ] #3 Windows admission metadata observations cost less per node with unchanged refusal verdicts, descriptor projections and fresh TokenOwner semantics where they matter
- [ ] #4 A fresh ChaChaNotes connection never leaves its journal_mode statement active, so retained frames cannot refuse a Send's commit
- [ ] #5 A Send attempt reads hook permissions once for its preparation consumers while effect gates and the final pre-dispatch check stay fresh
- [ ] #6 The live-turn transcript poll publishes progress without re-running the full config-locked reconciliation on every tick, and full reconciliation still runs on direct requests, periodically and at the end
- [ ] #7 Storage admission avoids repeated native re-walks inside one acquisition and between unchanged warm acquisitions on Windows, with full re-observation on any notified change, error or backstop expiry (ADR-126 amendment)
- [ ] #8 Warm and cold Send-to-provider-entry and first-feedback times are reported against the targets with the remaining unexplained delay
- [ ] #9 Any adopted overlap of independent stock configuration reads preserves initial MCP refusal, required versus optional failures, current-source validation, exact native retirement and custom-route ordering, and improves matched whole-Send timing.
- [ ] #10 Any adopted catalog-read consolidation removes measured redundant stock native proofs within each finite operation, preserves custom reader contracts, migration/effect/error handling, exact cancellation retirement and independent configuration/dispatch freshness, and improves matched whole-Send timing.
- [ ] #11 Raw-file append/rewrite performs one canonical file durability barrier before publication, retains the directory barrier and original failure/uncertainty handling, and records native work and sequential whole-Send evidence without weakening performance budgets.
- [ ] #12 One local capability manifest reads/parses installed server source once, retains fresh independent output and subsequent-call source/error observation, and reports caller-loop work separately from whole-Send latency.
- [ ] #13 Builtin-only stock composition omits unused external catalog reads/errors/audits under the approved ADR-225 refinement, including the stock fallback, while retained initial captures, mixed/unknown ceilings, required policy reads and actual invocation retain fresh gates; sequential measurements report the whole-Send outcome.
- [ ] #14 Context-only publication during an eligible manual Preparing attempt avoids redundant full reconciliation through the existing qualified route, while custom/legacy callbacks, direct FULL requests, source changes, deferral, teardown and native retirement retain their contracts; matched whole-Send evidence determines adoption.
- [ ] #15 First display synchronization of the unchanged chat and authored draft does not invalidate an accepted hook-review continuation; an actual chat/draft/permission change still refuses stale continuation and retains input.
- [ ] #16 Any adopted recovered-image publication route preserves actual catalog reads, transient/remote suppression release, existing recovery states, custom callbacks and pending FULL demand while avoiding unrelated reconciliation only within the existing live-poll eligibility; sequential whole-Send evidence determines adoption.
- [ ] #17 Stock context presentation invalidates on its actual held-compaction echo dependency, refuses an obsolete in-flight result, and avoids run-lifecycle-only reads; custom readers retain conservative lifecycle invalidation, existing ownership/freshness/action boundaries remain, and whole-Send timing is reported separately.
- [ ] #18 A stock character presentation refresh whose resident identity already differs avoids the redundant outer database precheck; unchanged/unknown/custom observations retain their fresh checks, the original refresh owns all reads and retirement, and work-count reduction is reported separately from sequential whole-Send timing.
- [ ] #19 Any adopted finite Windows SQLite preparation shares redundant parent work while preserving fresh pre-effect mutation checks, optional-generation bounds, final current-name association, exact native/admission retirement and custom routes; original work counts and sequential whole-Send comparisons determine adoption.
- [ ] #20 Any adopted finite MCP admission consolidates repeated source observations while preserving fresh proof before operation use/effects, current issued custody, custom-route contracts and exact native retirement; the explicit pre-lock refusal refinement follows ADR126, and original work counts plus sequential whole-Send evidence determine adoption.
- [ ] #21 Qualified stock owned capture avoids initial MCP policy/catalog reads and establishes a frozen ID/hash ceiling at each existing execution consumer's fresh composition under ADR225; explicit empty/custom/plugin contracts, invocation authority, saved-draft behavior and native retirement remain verified, with source-stable sequential whole-Send evidence.
- [ ] #22 Issued recovered-image native reads settle before screen/runtime and database teardown, including repeated cancellation; shutdown closes their existing admission first and never treats a timed-out drain or forced database close as retirement.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Repair the measurement harness; establish observer-free baselines at integration HEAD 36c6fb431a (sequential native runs only, guarded against overlapping runs).
2. Attribute the critical path with an out-of-repo all-thread stack sampler aligned to the probe's stage clock (never retaining frames).
3. Iteration 1: cut Windows native primitive cost (security descriptor read, conditional TokenOwner read, single node-tree build and component validation, qualification parse memo).
4. Harden the ChaChaNotes journal_mode statement (found while attributing).
5. Owner-approved levers, each implemented in its own worktree with adversarial review, then integrated and measured one at a time: hook snapshot per attempt, narrowed live poll, single path fence per acquisition, change-notified Windows evidence reuse.
6. Record receipts, ADR amendments, ledger rows and lessons.
7. Codex integration follow-through (user-transferred ownership, 2026-10-08): preserve the Claude candidate, run the three final-watch-gate race controls RED on it, apply the in-memory final-validity correction, then run those controls and the existing watch/native primitive/hook/poll tests sequentially. Keep native timing runs separate from all other native tests; reject detected overlap.
8. Re-measure the c504-based integrated candidate with matched cold/warm Send and rendered-feedback controls. Use saved pre-rebase measurements only as historical evidence. Attribute the remaining saved-to-trace delay before choosing another change; retain every deferred optimization.

9. OPT-31 bounded experiment: first add a real-body overlap/lifetime regression and establish serial-source RED. After hook review and original MCP entry checks, overlap its finite native preparation with the independent stock configuration producer. Keep the existing read handles and per-producer validation; publish only a complete joined snapshot. Preserve caller-loop custom callbacks, MCP-first failure precedence, optional empty maximum and repeated-cancellation draining. Run targeted lifetime/source/receipt controls, then quiet sequential comparison against 81c70f4a10. Adopt only with a measured gain; otherwise retain the serial implementation and record the rejected experiment.

10. OPT-28/OPT-73 bounded catalog experiment: follow `Docs/superpowers/plans/2026-10-08-console-owned-catalog-read.md`; establish original native count RED, share the original load body under the existing finite read owner, preserve custom/migration/publication gates, qualify focused controls and measure sequentially before adoption.

11. OPT-79 bounded cold-feedback cleanup (AC #8): move the existing question-validation import below the no-card/stale-card/staged-input early returns in `_answer_pending_question_with_draft`. Preserve the original validation and dispatch bodies. Prepare separately from the catalog candidate; qualify existing question routing/attachment controls and a natural cold feedback frame after catalog comparison so the measured changes remain distinguishable.

12. Qualify the combined source through ordinary DeepSeek setup and three saved turns; record actual overhead separately from stubbed/headless receipts. Attribute the post-save gap with the existing original-body observer in a real served Send before selecting another optimization. Isolate profile seeding and other concrete environment differences without changing product gates or extending test deadlines.

13. Publish parameter-free failed test identifiers and outcome counts from the existing targeted CI JUnit reports before artifact upload. Preserve test commands, deadlines and exit status; missing reports remain explicitly incomplete. Reproduce identified failures with the original targeted cases. This test-reporting change needs no new ADR.

14. OPT-81: investigate and repair missing qualification-file drive-root posture in Windows watch evidence. Establish a failing separate-content-drive control; retain the existing guard that refuses unwatched drive anchors and every native qualification/final root check. Same-drive evidence must not gain redundant paths. Qualify original watch/final-gate controls and the actual Windows CI path layout without weakening fast-path expectations.

15. OPT-80: reduce measured concurrent full observations of the same exact installed watch/evidence tuple. Use a finite per-watch serialization point outside the coordinator lock; only a waiting follower with freshly valid notifications, generation, expiry and drive roots may reuse the completed observation. Preserve original fallback, source/token/epoch/pause checks, independent final checks and retirement on failure. Establish original-source duplicate-work RED, then focused native success, invalidation, pause, exception and reentrancy controls. Measure sequential full-profile quiet before/after; retain only a whole-Send improvement. Missing/fresh-child/custom routes must preserve their contracts.

ADR required for steps 14-15: no new ADR.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md (TASK-34601 watch amendment) and backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: enforce the existing watched-dependency coverage and share only already-permitted fresh observations, without extending the backstop, caching authority, changing mutation epochs, or altering durable dispatch.

ADR required: no new ADR for the final-watch-gate correction, integration or finite catalog operation consolidation.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md and backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: restore the existing L4 final evidence-validity guarantee and verify implementations of already recorded architecture decisions. Any further policy change requires its own review before implementation.
16. Partition the current original configuration-capture boundary after the bridge diagnostic ruled out recurring prefix cost. Keep the original full-profile three-Send scenario and source-current, payload-free local checkpoints; identify actual repeated work before selecting another consolidation. Existing ADR-225 applies; diagnostic-only tooling adds no product API or policy.


17. OPT-86: establish an original-body RED for two flushes of one raw-file write despite no intervening write. Remove only the trailing os.fsync after the canonical flush_file and closed-wrapper bookkeeping; preserve every publication, native-close and uncertainty gate. Qualify real append and rotation, a second raw publisher, read-only zero-flush behavior, and canonical flush failure using isolated native ownership. Existing failure tests must target the canonical barrier rather than the deleted duplicate. Re-run the unchanged full-profile three-Send scenario sequentially with original budgets and compare against the frozen baseline; report actual native work reduction separately from whole-Send timing.

ADR required for step17: no new ADR.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md and backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: remove a redundant consecutive barrier while preserving the already-required canonical file/directory durability and finite resource ownership. No storage policy, cache, permission, transaction or ordering change is proposed.

18. Correct the incomplete Workspace availability fixture with its constructor-owned screen/read-lifetime state, and align the actual count cancellation test with the current wait-for-native-retirement contract. Retain held-worker, unpublished-result, refusal, retry and final-retirement assertions. Run the six identified cases sequentially before publishing; no production change or deadline increase. Existing ADR-225 governs lifetime behavior; test-only correction requires no new ADR.

19. OPT-89: the measured synchronous inventory takes21.7/23.2ms and its manifest rereads/reparses the same installed server source for tools, resources and prompts. Establish three-to-one original read/parse count RED; pass one invocation-local AST to the existing extractors. Keep direct helpers usable without arguments, fresh source on every manifest request, independent output dictionaries and original parse errors. No process cache or external catalog/permission change. Qualify existing tool/resource/prompt/schema controls and report work-count plus isolated caller-loop timing; do not present a noisy whole-Send sample as proof of the small saving.

ADR required for step19: no new ADR.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: implementation-local consolidation of three reads of the same installed source into one coherent manifest; existing public manifest schema, runtime and authority boundaries remain.

20. OPT-88: follow Docs/superpowers/plans/2026-10-09-console-builtin-only-preparation.md. Root owns shared preparation, integration and sequential native measurements; the controller/provider lane owns its consumers and integration tests; baseline verification reviews boundaries and evidence read-only. Record failing work-count/behavior controls before implementation, integrate both implementations before final checks, then compare against c223d2c91d sequentially with unchanged budgets.

ADR required for step20: yes, amendment to existing ADR-225.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md (Builtin-only composition refinement).
Reason: owner-approved change to unused external catalog error/audit semantics; no new ADR duplicates this domain contract.

21. OPT-10 context-publication experiment: follow Docs/superpowers/plans/2026-10-09-console-context-publication-routing.md. Reuse the existing source-qualified Preparing route for the default context memo publication, retain ordinary FULL fallback, and qualify real publication/ownership before quiet sequential measurement. No TTL/key/cache or durable-effect policy change.

ADR required for step21: no new ADR; bounded amendment to existing ADR-226.
ADR path: backlog/decisions/226-console-polling-and-full-state-reconciliation.md; preserves ADR-225.
Reason: reuse an existing qualified presentation route without a new boundary or weaker freshness/lifetime contract.


22. Revisit OPT28/73 as an initial-capture-only experiment after OPT88, following `Docs/superpowers/plans/2026-10-09-console-initial-catalog-read.md`. Establish original native count RED, preserve composition and all actual authority/effect/retirement gates, then adopt only after targeted controls and sequential matched whole-Send evidence. Existing ADR225/126 apply; no new policy or cross-module public interface.

23. Diagnose exact retained-source CI publication and hook-continuation failures using bounded original callback observations. Preserve successful publication, captured-owner refusal, draft retention, real review completion and native retirement. Correct fixtures only when evidence establishes an invalid route or startup precondition; keep original deadlines and performance budgets. Root alone runs local checks. Existing ADR225/226 apply; test/diagnostic corrections require no new ADR.

24. Establish a deterministic regression for the first same-owner draft refresh invalidating hook Send generation through the original callback. Preserve the initial callback/popup/indicator and real changed-owner invalidation. If the original regression fails, make the smallest correction at the existing callback boundary, then run original initial-draft and real review continuation/change/refusal controls. Do not claim this explains the historical intermittent failure without its exact race evidence.

ADR required for step24: no new ADR.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: restore the existing captured-owner continuation rule; no new authority, freshness, durability, storage, service or public callback contract.

25. OPT92 recovered-image publication experiment: follow Docs/superpowers/plans/2026-10-09-console-recovered-image-publication.md. Establish original-body routing RED with actual image catalog lookup and mounted publication; reuse the existing live-poll eligibility and exclusion/replay owners for this one disposable completion. Integrate image callback and screen wiring before targeted checks. Root alone runs tests and sequential interleaved whole-Send timing; adopt only with demonstrated benefit or an independently justified correctness improvement.

ADR required for step25: no new ADR; bounded amendment to ADR226.
ADR path: backlog/decisions/226-console-polling-and-full-state-reconciliation.md; preserves ADR225/126.
Reason: route one existing presentation completion through the existing live-poll contract without a cache, coordinator, native lifetime or authority change.
26. Repair the stale maintenance-control screen fixture exposed by step25. The unchanged a017 baseline reproduces its 45s active-case timeout: FULL now reads _current_console_attach_visit before the fixture's held retrieval boundary, but the stand-in omits it and waits blindly for an event after the task has already failed. Supply only the current owner/interface state and current FULL return contract, and surface early task exceptions instead of hanging. Preserve held-boundary, deadline, deferred demand, drain/reopen and torn-down assertions. No production lifetime change; run the six original scenarios. Existing ADR225/226 apply; test-only fixture correction requires no new ADR.
27. Correct OPT10 context presentation dependencies following Docs/superpowers/plans/2026-10-09-console-context-presentation-dependencies.md. Establish real held-resume and work-count RED before the two production edits, integrate distinct controller anchor and shared-key owners, then run targeted checks and sequential native comparison. Keep the image experiment separate and retain all negative/no-gain results.

ADR required for step27: no new ADR; dependency correction within ADR226.
ADR path: backlog/decisions/226-console-polling-and-full-state-reconciliation.md; preserves ADR225/126.
Reason: repair the existing disposable memo contract without a new owner, cache, TTL or authority policy.
28. OPT41 attribution before product selection: extend only the existing opt-in original-body Send observer to partition config companion admission into setup, member validation and completion, associating the numeric rows with the original hook read. Identify config/lock versus the three unused backup/temporary members without retaining paths, values or live frames. Require exact source/binding checks, bounded rows, paired original entries and retired local monitoring; preserve every native operation, test assertion and deadline. Root alone runs one sequential current-source full-profile diagnostic. This measures an inclusive opportunity ceiling, not a quiet speed comparison. If the unused-member cost is small, leave the broader writer/nested-cold contract unchanged and record the result. Shared/controller lanes perform only source review of explicit history ownership and existing finite-operation duplication.

ADR required: no new ADR.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: diagnostic-only attribution before choosing any new read or postcommit-order contract; no production behavior changes are authorized by this step.

29. Classify current storage-evidence re-observation before selecting OPT82/93 or a broader OPT41 contraction. Reuse the existing opt-in observer with four original-body targets: _reuse_evidence, _watch_current, _observe_evidence and resident DirectoryWatch.quiet. Record scalar original branch/count outcomes for fresh children, watch misses and full observation duration; do not invoke any extra source/native gate, retain file paths/objects or interpret a dirty notification as an unrelated change. Preserve exact source/binding checks, bounded rows and monitoring/native lifetime distinctions. Root alone runs one sequential full-profile diagnostic. Archive each one-off observer source with its receipt and restore the normal helper afterward; no production or timing-budget change. Existing ADR225 applies; no new ADR for diagnostic-only work.

30. Attribute the remaining admission fan-out to original calling domains after step29 shows165/129 acquisitions in warm Send windows but only two short full observations in each final hook. Extend that same one-off observer only with bounded code-location ancestry for _reuse_evidence and scalar task identity; retain no argument values, native objects or frames. Increase the4096-row ceiling to8192 because the prior receipt actually overflows after the third provider entry, preserving that earlier incomplete settlement result. Root runs one sequential full-profile diagnostic; distinguish work occurring during Send from its proven critical path. Use the result to select an existing domain owner for consolidation rather than a broad watcher/persistence policy change. Existing ADR225/226 apply; no new product contract or performance-budget change.

31. OPT91 bounded resident-identity refinement: follow Docs/superpowers/plans/2026-10-09-console-character-refresh-precheck.md. Establish the original two-callback changed-display control before the small product guard; reuse the existing stock reader qualification and original refresh owner. Controller lane owns character_context.py, verification lane owns only test_character_refresh_finite_batch.py, root owns integration/documentation and all sequential native checks/timing. Unknown/custom/equal ambient owners retain the precheck.

ADR required: no new ADR.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md and backlog/decisions/226-console-polling-and-full-state-reconciliation.md.
Reason: omit an already-determined presentation comparison before the unchanged complete refresh; no new cache, worker, freshness, service, authority or native lifetime contract.

32. OPT94 attribution: use a one-off bounded extension of the original local-event observer to distinguish CharactersRAGDB connection setup, journal-mode lookup, manual TRUNCATE checkpoint and native close. No SQL/body replacement, extra fetch, busy-result invention or retained native/frame objects. Root alone runs one sequential full-profile diagnostic on b5f9bc4c8f; record actual intervals and incomplete/unwind limits. Archive exact helper source/patch and restore after receipt. The observed15s connection timeout and14.723s saved-to-trace outlier motivate a hypothesis, not a causal conclusion. Existing ADR225 applies; diagnostic-only work needs no new ADR.

33. Follow the measured connection-setup lead (OPT95) with a bounded original-body diagnostic of private SQLite admission, trusted-parent verification, fixed-artifact preparation and the actual SQLite open. Keep policy/body/source checks unchanged, archive exact observer and receipt, and restore the temporary helper. Root alone owns sequential native runs; parallel lanes perform source review and offline analysis only. Select a bounded shared preparation contract only after attribution, with an ADR amendment and plan before product edits. No retained connections, permission cache, timeout change or checkpoint-policy change.

ADR required: no new ADR for this diagnostic.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: measurements only; any later change to shared native preparation must qualify its own contract.

34. OPT95 follows Docs/superpowers/plans/2026-10-09-finite-windows-sqlite-preparation.md with distinct product/basic-control/edge-control ownership and root-only integration/native execution. ADR required: yes; amend backlog/decisions/125-lock-safe-private-sqlite-validation.md before implementation, preserving ADR029/126/222. Source review resolves exact outcome/fallback APIs before edits; original regression precedes candidate application, then final integrated targeted and sequential quiet checks.

35. Partition the measured original MCP permission-load lifetime before choosing another implementation. The unchanged SQLite diagnostic's warm maximum capture is .587/.858s, of which permission read is .147/.681s; the slow read has .578s after raw-scope admission. Use bounded original-code/LINE observations to distinguish the owned body and lock, fresh checks, read/parse and retirement. Keep payloads, paths, receivers and frames out of retained records; preserve source-current, caps, retirement, saved turns and sequential native ownership. Source-only and offline lanes prepare/review the packet; root alone installs/runs/restores the diagnostic. Do not attribute an unlinked worker envelope to queue wait or sum nested spans as savings.

ADR required for step35: no new ADR.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: original-body diagnostic only, with no product boundary, cache, permission, source, lifetime, durability or timing-budget change. Any selected implementation still needs its own explicit plan and ADR check.

36. Follow the original permission diagnostic's slow source-validation envelope with bounded original control-record, registry, witness and final named-tree spans. Record current-thread CPU alongside wall time, plus process CPU at permission roots, to distinguish consumed CPU from elapsed waiting without assigning an unobserved scheduler, GIL, lock or disk cause. Preserve the same source/current-binding, caps, scalar-only records, generator lifetime, native retirement and sequential execution requirements. Run the original case once, archive the observer, restore it and analyze offline before selecting product changes.

ADR required for step36: no new ADR.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: diagnostic refinement only; no gate removal, evidence-cache eligibility, lifetime, permission, durability, timeout or product change.

37. Measure original Windows metadata primitive cost inside the existing positive tree-snapshot control before selecting an OPT90 refinement. Add optional scalar wall/current-thread-CPU aggregation to its existing original-code counter; retain all exact schedule, identity, source and physical-retirement assertions. Root runs only that node, archives/restores its temporary helper, and compares ntfs/info/security occupancy within the same original operation. No product edit or change to per-handle qualification is selected. Inherited volume proof is deferred because held-parent reparse ABA semantics remain unproved; a direct per-handle FileFsAttributeInformation query is only an unmeasured alternative.

ADR required for step37: no new ADR.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md; backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: original native cost observation only. A selected volume-check or ownership change would require its own explicit ADR check and qualification.

38. Follow Docs/superpowers/plans/2026-10-09-finite-mcp-source-admission.md with shared product, controller boundary-control and baseline original-count ownership. Root owns integration, original regression and all sequential native verification. Amend ADR126 before implementation; preserve ADR225. Consolidate the whole stock admission boundary rather than silently deleting a guard or adding a witness cache. Final source/parent proof precedes usable operation publication; custom callbacks retain their route. Integrated targeted controls precede sequential quiet ABBA comparisons and adoption.

ADR required for step38: yes, amendment.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: explicitly revise pre-lock refusal ordering while preserving the before-effect proof and lifetime contract.

39. Follow Docs/superpowers/plans/2026-10-09-console-composition-tool-ceiling.md under ADR225. Shared lane owns detached mode/owned capture, integration lane owns controller/provider publication, baseline lane owns new original-body and boundary tests. Root owns plan review, original RED, integration, all targeted native checks and final sequential ABBA. Explicitly move the ceiling to each existing execution consumer; do not add a versioned cache or silently reuse None/empty semantics.

40. Correct the existing recovered-image shutdown boundary under ADR225/226. Close admission synchronously, capture/cancel the exact image-owner tasks and retain their actual completion before screen/runtime and file-backed database teardown, including repeated cancellation. Reuse the existing Textual WorkerManager and add the image group to existing view-worker capture so both current and detached views drain. Retain the exact Task in the existing image set and preserve nonfatal unexpected-error policy, native-read custody and original timeout/refusal semantics. Wire ordinary app, Screen and test-host boundaries without a new registry. Establish focused real held-read shutdown controls and rerun the strict Close retirement case. Root alone runs local native checks. This is required correctness closeout, with no latency optimization, forced DB close or new task registry.

ADR required for step40: no new ADR.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md; backlog/decisions/226-console-polling-and-full-state-reconciliation.md.
Reason: direct implementation of existing finite-read retirement at the existing image owner and host shutdown boundaries.

<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Trace → fix → re-measure, one attributable change at a time, on native Windows with the full app (private profile, file-backed DBs, only the provider adapter stubbed). Latency claims come only from observer-free, guarded (no overlapping native run), interleaved A/B runs of `Tests/Performance/test_console_send_wall_clock.py`; attribution came from an out-of-repo all-thread stack sampler aligned to the probe's stage clock.

- **Harness:** both Send probes wrapped `accept_received_intent(intent)` and crashed after `6cad32ed67` added `_configuration_preparation`; they now forward keyword arguments.
- **Iteration 1 (`Utils/windows_files.py`, `Backup_Recovery/qualification.py`):** `security()` uses `NtQuerySecurityObject` (GetSecurityInfo silently re-read the parent's descriptor for every app-created directory); TokenOwner is read only when it can change the projection; the admission snapshot builds its node tree and validates components once; the qualification JSON parse is memoized by exact text. 40-node snapshot 10.2–10.9 → 4.8–5.1 ms; warm Send 6.0 → 4.5 s.
- **PRAGMA hardening (`DB/ChaChaNotes_DB.py`):** the journal_mode statement is fetched. A retained cursor (any frame-retaining tool — here the first sampler) made every later commit fail with "SQL statements in progress".
- **L1 hook read per attempt** (`Agents/hook_permissions.py`, `Agents/run_hooks.py`, `Chat/console_*`): one full hook-permission read serves the pre-commit preparation consumers; effect gates and the final pre-dispatch admission stay fresh; mismatched context falls back to a fresh read. Warm −0.76 s, cold −1.35 s.
- **L2 narrowed live poll** (`UI/Console_Modules/poll_cadence.py`, `UI/Screens/chat_screen.py`): light publication ticks; full reconciliation on direct requests, ≤2 s and at the stop tick. No Send-latency change; fewer UI stalls. Rebased onto #3023, whose own Preparing-poll narrowing (`_sync_console_poll_display_ui`) now serves every poll-driven full pass; only complete (not Preparing-narrowed) passes reset the 2 s cadence.
- **L3 one path fence per acquisition** and **L4 change-notified Windows evidence** (`Backup_Recovery/storage_admission.py`, `Utils/windows_files.py`): see the ADR-126 TASK-34601 amendment. Native opens per warm Send 37.9k → ~20k; the native pause probe's open budget passes again.
- Docs: ADR-126 amendments (iteration 1; L3/L4), ADR-225 note (L1), User Guide cadence note (L2), ledger rows, lessons (sampler frame retention, `monkeypatch.undo`, by-id watch handles, shared `git stash`).

Pre-existing failures observed identically on unmodified HEAD (not caused here): 10 in `test_participant_lifetimes.py`, 4 hook tests (`test_console_run_hooks_regressions` ×2, `test_hook_permissions` unsafe-store and v2-revoke), `test_console_presentation_cadence::test_cancelled_actual_count_read_drains_on_its_receiver_and_retries`, flaky `test_admission::test_known_incompatible_is_refused_until_os_lifetime_ends`; symlink tests need a privilege this host lacks.

- **Codex final integration (PR #3049 at bdff2a2d95):** a locked final watch-validity recheck now covers notification-silent in-process writes, expiry and explicit invalidation after lease counting/root observation. Three native controls fail on the original and pass with the correction; existing/new watch tests 21 passed, 1 owner-privilege skip; combined primitive/snapshot/WAL/hook/poll checks 92 passed. ADR-126 already requires this validity; no new policy or native read was added.
- **Fresh final-source comparison:** sequential full-app ABBA on c504 plus the probe-only fix versus PR #3049 plus this correction, two fresh processes and three Sends per arm. Warm mean 7.809 -> 3.429 s (candidate 3.301-3.744); cold mean 8.324 -> 5.460 s; Send-period >100 ms heartbeat intervals 87 -> 16. All 12 Sends persist and settle correctly, source fingerprints stable, no detected native overlap. Evidence: task artifact `claude-watch-final-gate-review/integrated-comparison.json` and four raw run directories. Small diagnostic sample; no physical-frame claim. The one-second target and cold-feedback qualification remain open, so this task remains In Progress.
- **OPT-31 evaluated and rejected:** real-body overlap and 107 targeted cases qualified the experiment, but quiet ABBA warm mean was 3.255 s serial versus 3.332 s parallel, cold 5.355 versus 6.153 s. Source/handles/persistence remained correct. Restored the simpler serial implementation; candidate, corrected controls and receipts are retained in `claude-watch-final-gate-review/overlap-candidate` and `overlap-comparison.json`. No additional coordinator or task lifetime is shipped.
- **OPT-28/73 evaluated and rejected:** 45 targeted controls passed, but the initial ABBA catalog gain did not survive the final BA confirmation (candidate warm 4.273 s versus unchanged 3.245 s). Archived exact implementation/tests and restored the simpler source. Queue attribution separately shows preparation waits under 10 ms, with actual bodies taking hundreds of milliseconds; no executor change is justified. See the optimization ledger and catalog experiment plan for receipts and limitations.
- **OPT-79 retained:** ordinary Send skips the question-validation import when no live question can consume its draft. All 18 targeted routing/app checks pass after two stale fixtures were updated to the current Send wrapper. Fresh natural Preparing frames were 85.877/21.035/30.115 ms; all three turns persisted and settled. These headless observations do not qualify physical terminal latency or explain earlier slow frames. Added the routing selection to the three-platform CI job; remote results are pending.
- **Final integrated delta checks:** final-integration-2 passes all nine watch-validity, actual Enter/button message-pump, exact-owner, repeated-cancellation and final-poll-completion cases on the frozen combined source (61.34 s). Earlier 92-case integration remains applicable to unchanged native/hook/WAL/poll components. OPT79 adds no lint diagnostics: ChatScreen retains 43 existing findings; its test file has zero. The optimization ledger preserves all 79 candidates, including both rejected experiments.
- **Real combined-source UAT:** dd80a40dfb40 completes three non-streaming DeepSeek/deepseek-chat turns via Enter/button/Enter with context retained, six saved messages, three trace links and zero pending checkpoints. Original sources and owner config remain unchanged. App/server tree retires and native identity releases without forced termination. Action-to-adapter is 11.812/7.343/8.422 s, including 7.000/2.719/4.328 s after save and before trace reservation. The actual-use latency target is still unmet; stubbed/headless timing cannot stand in for this. The prior connection failure occurred before any Send and remains preserved/excluded. Environment shape and real-path attribution are now the next diagnostic; no new product optimization is justified yet. See the optimization ledger and task artifact real-uat-summary.json.

- **Current warm attribution:** a source-frozen original-import diagnostic completed three persisted turns, with no detected native-run overlap or monitor gaps. Imports occupied a disjoint 1.698 s on the first Send, 18.540 ms on the second and 0.018 ms on the third. The global unwind observer added unmeasured overhead; its whole-Send times are not speed evidence. Warm multi-second delay lies outside observed imports.
- **Profile shape control:** sequential full-default versus original-minimal profile runs were 8.955/6.657/6.973 s versus 10.677/7.054/7.322 s to the provider adapter. All six turns persisted and settled with unchanged sources and no detected overlap. This single pair does not support config file size as the explanation for the slow real-provider run; it cannot establish a causal percentage or a general speed comparison.
- **Real diagnostic:** one further actual DeepSeek reply completed and persisted on dd80 with normal process retirement. The saved-to-trace interval was approximately 2.250 s; disjoint named preparation/native-entry spans covered approximately 1.744 s of it. Permission-owner construction and an uninstrumented bridge setup interval remain candidates. Windows Python 3.12 monotonic stage logging has 15.625 ms resolution, while observer spans use QPC; alignment is approximate and includes an unmeasured body-entry-to-stage interval. This preimported diagnostic is not a cold-speed measurement.
- **CI qualification remains open:** the exact failed Windows raw-pin selection passes locally (65 passed, 9 platform/privilege skips, 31.13 s). This does not clear failures on the CI host or its privilege-only cases. Failed remote jobs currently expose only exit codes; two relevant jobs now receive identifier-only JUnit summaries so the failure sets can be investigated without publishing assertion details.


- **OPT-81:** preserve the existing all-drive watch guard by including the installed qualification file's Windows drive root in selector posture; same-drive set deduplication and POSIX behavior remain unchanged. Real source D:/private profile C: regression failed before the change and passed with it (22 passed/one privilege skip). Final retained-source native selection: 65 passed, ten expected platform/privilege/layout skips. Tests count actual stamped roots and assert original native handles retire. Existing ADR-126/222 apply; no new policy.
- **OPT-80:** tested concurrent observation sharing, including event-loop nonblocking fallback and affected-caller revalidation. Original three-observation RED became one; ten candidate controls and 31 combined watch cases passed. Sequential full-profile ABBA did not justify adoption: warm baseline 4.307307 s versus candidate 5.766554 s; both had 46 Send-period stalls above 100 ms. Removed the candidate and retained its exact tested source, controls and all receipts outside the checkout. No claim that this small sample proves a causal slowdown. See the optimization ledger for raw samples, initial failures and comparison limits.
- Current quiet stage partitions keep initial hook/config preparation and saved-to-trace orchestration as the major warm costs (78–89% together); bridge-setup checkpoint attribution remains pending. One-second application overhead and consistently sub-100-ms rendered/input feedback are not achieved; task remains In Progress.

- Final current-source Console consumer checks passed all 85 cases in 535.28 s: hook consent sharing and native lifetime, original polling/reconciliation/cadence, tab-source changes, and full configuration-sync refusal/lifetime. No test deadline changed. Combined with the 65 native passes, this is 150 final targeted passes with ten expected native-only skips. Receipt: watch-final-consumers-1. One owned-process stack capture during verification confirmed real app tests were progressing behind buffered output; it was not a Send timing run.

- Original bridge setup attribution on 9936f409c6 is complete: .468188 s cold, .000563/.000386 s warm, dominated only on cold by plugin-service lookup (.403024 s). All three saved Sends/traces complete; source/checkpoint/monitor custody passes. Warm saved-to-trace remains1.410/1.455 s. OPT-84 retains the cold-only alternative; current diagnosis returns to initial configuration capture, whose historical inclusive costs must be refreshed before new product work.

- Recovered CI logs identify the Linux Preparing failures as direct fixture reads of the lazily created _console_control_bar_replay_whole_sync flag (three cases). Both fixture reads now use the original product's absent=False contract. Four affected real Windows polling scenarios pass in55.22s, with lint/format checks passing and all original deadlines preserved. The remote Windows log has one failed membership precondition before the deliberate mutation: the initial FULL did not reach the held Surface lock within10s. Its cause remains unproved; local final85-pass coverage does not clear that remote observation. Existing step13 covers the fixture correction; no new ADR or product policy.


- **Current capture attribution:** capture-final-1 and narrower capture-domain-1 complete original-body/source/retirement checks. Warm prompt inputs cost only53/64ms (world10/13ms); OPT85 finite Notes-read grouping is retained as a deferred source-only plan, without code or tests. The ledger now keeps85 IDs. Warm saved-to-trace remains1.726/1.414s in the narrower run; current original postcommit attribution is next. A new quiet baseline at5ec3 completes4.885/3.139/3.495s Sends with all saved replies/traces settled; no speed or physical-feedback acceptance follows.
- **Exact-head CI:** run37891900210 now passes raw storage/watch and Preparing selections on all three hosts; Windows qualifies actual D-source/C-profile layout (68 passes/seven skips), and each Preparing job passes38+18+47 targeted cases. Startup still breaches original time/heartbeat budgets. Trace diagnostic settles all three replies and native custody but fails Send1 POSIX helper count38>16; all-thread attribution and introducedness remain unproved. No budget change, full-suite run or task completion.


- **OPT86 redundant raw-file barrier removed:** actual source-owned append/rewrite proves two native flushes before and one after; reads stay zero. Ten distinct targeted durability/publication/cancellation/refusal controls pass, including isolated canonical-flush failure that retains exact uncertain native custody and blocks maintenance until process exit. First canonical barrier and directory durability remain unchanged (ADRs126/222). Corrected platform-inaccurate flush spies and Windows TOML fixture escaping; seven cases added to existing three-platform CI. Quiet ABBA warm means3.698793s baseline versus3.720342s candidate demonstrate no whole-Send gain. This deletion is retained as simpler, less redundant work, not a successful latency iteration; the subsecond target remains open. Receipts: single-flush-{red-1,green-1,controls-1,controls-2,a1,b1,b2,a2}, single-flush-comparison.json. The ledger now retains88 potential optimizations and15 follow-ups, including unimplemented history-selection and builtin-only-catalog ideas.

- **2026-10-09 follow-through:** current original-body warm tool preparation is .416/.525s, including an empty external catalog read of .264/.378s. OPT88 read omission would change unrelated audit and catalog-error behavior; it is not adopted. Original Ubuntu CI also fails helper starts33>16, so this is not diagnostic-only; budgets stay unchanged. Detailed receipts and initial fixture-correction failures remain in the optimization ledger. The six identified fixture/cancellation cases now pass in10.87s (fixture-final-integrated-1), with source stability, original deadlines and lint/format verified. Registry/database replacement asserts no cache publication; held cancellation checks custody before release and retirement afterward. No product change or latency acceptance is claimed. Cross-platform requalification remains pending.
- **OPT89:** one fresh installed-source parse now builds all three builtin manifest sections; direct helpers, subsequent source/error observation and independent output remain. Final targeted17passes, unchanged existing lint findings, independent review clear. Isolated manifest16.557ms→6.388ms is one descriptive sample. Quiet sequential ABBA warm3.531→3.727s shows no whole-Send gain; all12 turns persist/settle and source/overlap checks pass. Retained as a small preparation simplification under ADR-225, not elapsed-time acceptance. The deferred ledger now retains90IDs. Existing cross-platform Preparing workflow includes the eight relevant manifest cases; remote qualification of this new product revision remains pending.
- OPT92 image publication was qualified then archived without adoption: stable A2/B1+B2 warm means4.700/4.731s are tied; a favorable BA confirmation and slow A1/A3 make the aggregate inconclusive. All six samples and original/negative controls remain in the experiment plan. Product/candidate-only tests restored; no new route ships. The exposed maintenance fixture repair remains (six passing controls, unchanged deadlines); original baseline independently reproduces its missing-seam timeout. Existing ADR226 records disposition. Context dependency correction starts separately under AC17.
- OPT10 context dependency correction: exact stock presentation keys on the real held echo; custom/subclass/replaced sources keep tagged lifecycle invalidation. Six causal original failures,57 final integrated passes and2 headless frame-budget passes qualify stale-result refusal, real resume, ownership/config fences and native retirement. Original version-read reuse is2-to1 for lifecycle-only changes. Quiet warm means4.108-to6.057s are worse amid strong baseline variation; no speed gain claimed. Retained for the independent correctness defect, with full samples/limits in Docs/superpowers/plans/2026-10-09-console-context-presentation-dependencies.md and ADR226. No TTL, performance-budget, permission or saved-dispatch policy change. The main latency target remains open.

OPT94 diagnostic sqlite-close-detail-1 completes on b5f9bc4c8f: 1 passed,
three saved/settled turns, 42 stages, unchanged source/HEAD and no detected
native overlap. All 596 selected SQLite rows finish with zero overflow,
unmatched or unfinished rows; original bindings and monitoring retirement pass.
Pre-provider checkpoint counts 19/16/15 take 47.795/43.559/36.656ms inclusive;
warm saved-to-trace checkpoints total only 10.881/11.621ms. This run does not
reproduce a long checkpoint stall or explain the earlier 29s outlier. OPT94 is
deferred rather than changing retirement policy on this evidence. Fresh
connection setup is larger: 20/16/15 branches and 1.353/.541/.952s inclusive,
with setup journal queries .175/.213/.139s. Concurrent/nested intervals are not
additive savings. OPT95 investigates the original connection preparation.
Exact observer/patch and attribution-summary.json/md are archived beside the
receipt; helper SHA256 cad7b899319329ae464e7237d4543ee123b31b60e926e3781ab8bf942405513a.
The tracked helper is restored; the overall latency target remains unmet.


OPT95 original setup diagnostic sqlite-setup-detail-1 completes with three saved
turns/42 stages, unchanged source and no detected overlap. All888 setup rows pair;
zero caps/unmatched/unfinished/error-cleanup markers; original bindings/source
and monitoring retirement pass. The separate raw ancestry observer retains92
misses. Warm24/17 setup operations have admission prefixes .2025/.0992s,
directory verification .2569/.1572s, main preparation .4076/.0857s, sidecars
.7783/.2656s and actual SQLite connect .0295/.0106s. Components partition each
operation; operations can overlap, so these totals are not Send savings.
The actual SQLite open is under2% of setup work; fixed artifact preparation is
55.8–69.9% and directory verification another15.1–24.9%. Warm saved-to-trace
has8/5 setups and .3322/.0975s artifact preparation. The earlier29s outlier is
still unexplained. Exact observer47a7cffb69565c08ff31f0ad2a83fc8b28c0c3d0a91db07a2c5cf4e3e963e695,
patch and attribution-summary.json/md are archived beside the receipt; tracked
observer restored. This supports the bounded fixed-inventory preparation
experiment under the OPT95 plan/ADR125 amendment, not a connection pool,
checkpoint-policy change or timeout adjustment. Product qualification is pending.


- Step37 original metadata control passed1/1 in1.10s, preserving ten-node native
  schedule, real handle retirement and source assertions. Snapshot5.973ms;
  twenty ntfs bodies0.433ms inclusive. No volume primitive rewrite selected.
  Observer123/123 paired, no caps/unmatched/unfinished; original helper restored.
  Detailed receipt/caveats in the optimization ledger. This is not whole-Send
  qualification; the overall latency target remains open.

- **Recovered-image shutdown correctness (AC22):** both original-product current/detached lifecycle controls fail their final sealed-admission and cancelled-caller-waited facts after the actual payload read and owner cleanup (`recovered-image-shutdown-red-1`, 073f3). The reviewed correction uses the existing Textual WorkerManager/image task set and joins actual completion before runtime teardown, including repeated caller cancellation; unexpected image errors remain nonfatal. Source/static checks pass, but native GREEN and current-head CI remain pending. The actual Linux CharactersRAGDB Close refusal is retained; this source-proven image-owner gap does not by itself identify that CI operation's issuer. No forced database close, new registry or deadline change is introduced.

<!-- SECTION:NOTES:END -->

- **2026-10-09 OPT-88:** implemented the user-approved ADR-225 builtin-only composition refinement in shared MCP preparation and the stock controller/provider fallback. One pure dependency rule, defining-source qualification and an issued-result completeness flag remove the unused external read without changing initial capture, common policy, custom paths or invocation gates. Added native regression controls;255 distinct cases qualified across the initial248-pass run and corrected12-case rerun. Full-profile sequential ABBA warm mean3.742618→3.398968s (~9.18%), cold5.334600→4.497071s; all12 turns persisted/settled, no detected overlap or source change. The ledger records raw samples, test-assumption corrections, remaining platform failures and limitations. Existing ADR-225 amendment and Docs/superpowers/plans/2026-10-09-console-builtin-only-preparation.md govern this behavior change. Overall task remains In Progress; one-second application and physical-feedback targets are not achieved.


- **OPT10 bounded context publication (ADR226):** changed context results use the existing qualified manual-Preparing display route, preserving captured custom/legacy FULL callbacks and trailing demand during overlap. Original routing RED2F4P and a direct-swap negative control establish the avoided work and overlap hazard. Final integrated51pass; strengthened publication6pass; lint/format and independent review pass. Sequential ABBA plus BA completes18 saved turns with current sources/no detected overlap. Confirmation warm mean3.014297→2.799411s; all six candidate warm samples2.731259–2.873153s. The15.515540s baseline outlier remains in evidence; do not claim the distorted56.3% mean difference. See Docs/superpowers/plans/2026-10-09-console-context-publication-routing.md and the optimization ledger. Native owners, keys/TTL, budgets and live permission/durability gates stay unchanged. Exact final-source remote and physical-terminal qualification remain open; task stays In Progress.
- **OPT88 CI follow-through:** c9fb3fbc6a corrects two obsolete empty-ceiling external-read assertions while retaining provider construction and unavailable-plugin refusal. Native correction/neighbor selection13pass; both rerun successfully in the final integration selection. Broader CI failures are retained for separate diagnosis.


- **Initial-only catalog experiment:** rejected after original work-count RED,
  79 qualified candidate controls and sequential ABBA/BA. Initial ABBA warm gain
  is only 50.37 ms (1.93%); a later broad baseline slowdown is unexplained.
  Approximately 213 additional product lines are not justified. Restored the
  retained implementation and archived exact candidate source, tests and receipts.
  Existing ADR-225/126 and the initial-catalog plan document the decision.
- **Same-owner continuation and publication follow-through:** initial display
  synchronization no longer cancels the hook generation when the chat owner is
  unchanged; real owner changes still cancel. Two causal original-source failures
  and the successor control establish the boundary. The publication fixture now
  awaits completed FULL reconciliation and its actual publishers within the
  existing budget; reader-drift cleanup preserves the primary failure. All 54
  final integrated controls pass in 416.97 s with stable source, lint/format and
  independent review clear. Modified wiring.py and the three focused UI tests;
  ADR-225 applies, with no new policy or public interface. This does not prove
  the historical intermittent CI refusal's cause or satisfy the latency targets.
  The ledger retains exact diagnostic limits, current post-save attribution and
  the ruled-out duplicate run-log investigation. Task remains In Progress.

- **Current real-provider check (a3f5ba2517):** three actual DeepSeek Chat replies
  retain context and save six messages/three complete trace links with no pending
  checkpoint. Product/Test source, HEAD and original config stay unchanged;
  normal App/server shutdown proves empty owned tree and released native identity.
  Action-to-provider is5.359/3.547/3.719s; warm saved-to-trace1.657/1.563s.
  This is functional qualification, not a causal comparison or physical-frame
  measurement. Third collapsed-paste Enter did not dispatch; expansion plus Send
  completed it. FOLLOWUP16 records this separately. Completed bae CI still exceeds
  Windows native-open and POSIX helper budgets; no budget/deadline was changed.

- **Current attribution follow-through:** companion-member loop is not exercised in
  the ordinary profile; watch misses do not justify broader cache/invalidation
  changes. The final bounded caller receipt completes three saved turns with
  zero watch overflow/unmatched/unfinished; ancestry misses/truncation remain
  explicit. Warm Send2.903/3.162s is diagnostic only. Connection-creation entry
  repeats across independent domains, but retaining worker connections lacks an
  App shutdown owner. Existing OPT91 finite character-display grouping is being
  reviewed; no new product optimization or speed gain is claimed. All one-off
  observer sources/patches and analyses are archived with their receipts, normal
  helper restored. Ledger preserves93 IDs and the rejected/unselected options.

- **OPT91 resident display precheck:** qualified stock resident mismatch now
  enters the existing complete refresh directly. Original RED and59 distinct
  targeted passes preserve revision/custom/error/source/owner/cancellation and
  retirement behavior; one stale retry-count assertion was corrected with exact
  per-attempt reads. Changed display2→1 callbacks and663→405 native opens; direct
  action unchanged. Quiet ABBA warm4.553→10.964s includes an unexplained29.089s
  candidate turn and establishes no whole-Send gain. Retained on this draft
  branch for proven work reduction, with performance uncertainty explicit;
  latency task remains In Progress. Plan: console-character-refresh-precheck.md;
  existing ADR225/226 apply. Ledger now retains94 candidates, including distinct
  unimplemented OPT94 finite-close WAL checkpoint housekeeping.

- **OPT94/95 measured disposition:** original close attribution did not reproduce
  a long checkpoint stall; checkpoint policy remains unchanged. The fixed SQLite
  artifact experiment halves parent walks and removes 16 native opens per setup,
  with 55 distinct targeted cases qualified after correcting five Windows edge
  fixtures; six owner-privilege cases remain unqualified. Quiet ABBA warm means
  3.030 to 3.224s provide no whole-Send benefit, so the additional product/test
  machinery was archived and removed. Cold means 5.228 to 4.941s and all twelve
  saved/settled turns are retained in the result. Source unchanged within each
  run, no detected overlap, only the intended two normalized product differences.
  Existing product lint/format debt remains explicit; new test lint/format pass.
  ADR125 records the rejected alternative, preserving ADR029/126/222. Plan and
  complete evidence: finite-windows-sqlite-preparation.md. Ledger has 96 IDs,
  including the deferred application-owned connection-worker alternative; no
  retention, checkpoint or durability change is selected. Task stays In Progress.

- **Original permission partition:** permission-load-detail-1 passes with three
  saved turns, 42 stages, stable sources and no detected native overlap. All six
  permission bodies take the inactive/unreadable-or-missing default branch;
  no permission file read or JSON parse occurs. Warm initial permission roots
  take .119/.153s; composition roots .146/.488s. The slow warm final raw check
  takes .356s, including .004s parent proof and .352s before that proof. Cleanup
  is .10-.59ms and mutation-fence acquisition 9-27us across the six roots.
  These observed intervals do not establish CPU, disk, scheduling or lock causes.
  All 50 permission rows complete with zero caps/unmatched/unfinished; 200
  ancestry misses, seven unclosed base starts and 92 separate hook/raw misses
  remain explicit. Observer archived/restored. Step36 partitions the shared
  recovery proof with CPU and wall clocks; no product optimization is selected.

- **Recovery-proof CPU partition:** permission-source-detail-1 passes three saved
  turns with stable source/no detected overlap. All six loads still take defaults;
  42 full witness/control observations occur (12 cold initial, then six per root).
  Each warm load has four observations in admission, one in readable/default
  handling and one final check. Initial preparation and final named-tree work
  dominate; records themselves are small. Warm initial roots use .203/.203s
  measured thread CPU in .244/.261s wall; composition .047/.141s in .150/.202s.
  GetThreadTimes values fall on a15.625ms lattice; individual short CPU deltas
  are not precise. The prior long raw-check outliers do not recur. All276 rows
  pair with zero caps/unmatched/unfinished;562 ancestry misses, seven unclosed
  base starts and92 separate hook/raw misses remain explicit. Observer restored.
  Step37 measures the original shared native primitives before another edit.


- **OPT76 disposition:**86 distinct targeted cases qualified the explicit finite
  admission refinement; empty permission-scope work fell4→2 full witnesses and
  349→208 native opens. Quiet sequential ABBA warm means4.534→5.691s show no
  whole-Send gain amid substantial variation, so118 additional product lines
  and candidate-only test migrations were archived and removed. All12 turns
  persisted/settled with stable source/no detected overlap. One source-drift
  receipt is excluded; final mutation/retirement control passes unchanged-source.
  ADR126 and finite-mcp-source-admission.md retain the full decision/evidence.
  Overall latency and physical-feedback targets remain unmet; task stays In Progress.


- **OPT98 retained:** stock owned capture now leaves definition selection to the
  existing execution consumer, removing initial policy/catalog reads1/1 to0/0.
  Fresh composition and invocation gates remain. Root/peer review and259 distinct
  targeted cases qualify source replacement, cancellation/retirement, legacy and
  plugin ceilings, durable ordering and retained drafts. Three old initial-read
  tests were explicitly migrated; real held-Workspace lifetime proof remains.
  Quiet full-profile ABBA warm3.031 to2.718s (10.34% lower), cold4.885 to3.945s,
  all12 turns saved/settled with source/no-overlap checks. Initial a1 excluded for
  external evidence-reader overlap, replaced by a3 before candidates. Four product
  files, three test files and existing native CI wiring changed. ADR225 and
  composition-tool-ceiling plan record the deliberate observation-boundary shift.
  OPT99 retains the deferred versioned metadata owner;99 IDs remain in the ledger.
  Whole-Send one-second and actual-terminal100ms targets remain open.


- **Committed OPT98 follow-through:** two current Enter/button held-reader cases
  pass with Preparing34.671/41.205ms and input frames26.718/9.178ms, qualifying261
  distinct targeted cases in total. These are headless frames. The separate
  current post-save diagnostic completes three saved turns with stable source
  and no detected overlap; composition, final hook checks, history, checkpoint
  and run startup each remain real costs. No further whole-stage duplicate is
  established. Existing observer gaps and exact intervals are recorded in the
  composition plan and ledger. No new product change or current native CI pass
  is claimed; the overall task remains In Progress.
