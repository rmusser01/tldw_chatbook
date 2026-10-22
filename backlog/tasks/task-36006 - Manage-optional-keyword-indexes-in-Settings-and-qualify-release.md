---
id: TASK-36006
title: Manage optional keyword indexes in Settings and qualify release
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 14:32'
updated_date: '2026-10-22 03:17'
labels: []
dependencies:
  - TASK-36005
documentation:
  - Docs/superpowers/plans/2026-10-20-workspace-project-keyword-index.md
  - backlog/decisions/240-workspace-project-context-and-folder-bound-index.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users explicitly manage folder-bound keyword indexing and understand coverage from Settings and Console.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings exposes truthful Off/Building/Partial/Ready/Paused/Interrupted/Error controls and processing choices using existing UI tokens; Console shows status and a Settings route.
- [x] #2 Compose/status projections perform no I/O, stale actions have no effects, and late binding/job/epoch callbacks cannot overwrite another view.
- [x] #3 Isolated app journeys cover build/query/pause/resume/restart/refresh/detach/Clear with preserved Library choices, no semantic controls, and a real production query pin that holds maintenance/restore replacement until physical retirement; isolated restore/rollback invalidates old scope and key.
- [x] #4 Final service/store/tool qualification covers at least20,000 text files across20,000 nested/sibling folders plus a separate20,000-file single directory, physical retirement, permission races, missing embeddings and blocked sockets, with measured limits and updated architecture/user documentation.
- [x] #5 Multi-directory keyword units preserve aggregate256-entry/256-directory-attempt/256-raw-inspection bounds and existing source/frame/read budgets, fresh unit admission plus atomic final publication, idempotent completion/pruning and explicit recovery; durable indexed frontier state replaces tree-sized visited memory, with compatible cache migration and measured small-phase improvement before literal scale qualification.
- [x] #6 Keyword search can discover a unique eligible filename or path term absent from its body, including empty/whitespace-only eligible text files, while excluded/changed/deleted files cannot leak names or cached text; existing v2 content is migrated atomically within capacity and final path+text latency is qualified.
- [x] #7 Processing controls default to all otherwise eligible UTF8 text, preserve saved empty/partly empty policy choices as all, support explicit narrowing, and management remains able to Clear a validated legacy cache under migration capacity pressure without silently upgrading or blocking the UI during close.
- [x] #8 Index close drains already-issued management/query/preparation operations through actual outer native registry/cache retirement, rejects later admission and remains responsive under repeated waiter cancellation; unproven closure is retained and reported.
- [x] #9 Detach immediately revokes index access/publication and leaves a supported durable cleanup-only retry after restart, without reattaching source authority or touching a replacement binding at the same path; failed cleanup remains truthful.
- [x] #10 Included-folder comparisons agree between actual scanning and public keyword search for case/Unicode variants and preserve exact peer-prefix, containment and exclusion boundaries.
- [x] #11 Committed compatible Building coverage is discoverable through Console preparation and public search with truthful partial results, preserving scope/manifest/live-source/revocation gates.
- [x] #12 Proven-native-retired run/error history is bounded without evicting unresolved native owners, public scanner annotations describe compatibility/unit shapes, and status reloads preserve unsaved processing inputs. Qualification: full still-running-supervisor pruning is deferred as accepted nonblocking Minor M1; final-re-review.md records the exact physical-thread tail limitation, not an all-seven closure.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read approved K5 brief, source-verified UI seams, design language/component patterns and Impeccable product context. Write mounted regressions before UI/service changes.
2. Add exact selected-binding management card and narrow admitted immutable service policy projection; no constructor/compose/event-loop I/O. Preserve scope/job CAS and mounted binding/epoch callbacks, explicit build/resume, truthful busy/partial/errors, existing tokens and navigation. Console projects status and routes to Workspaces Settings after safe modal dismissal.
3. Run targeted UI/policy GREEN and covering files. Qualify isolated app build/query/pause/resume/restart/refresh/detach/Clear, all saved Library choices, real production query storage pin through actual native retirement and isolated encrypted restore/rollback invalidation.
4. Qualify final service/store/tool path on at least20,000 text files across20,000 nested/sibling folders plus a separate20,000-file single directory. Retain stable private corpora, real marker response, machine/corpus/throughput/baseline+peak-extraRSS/whole-binding disk and sidecars/cancellation retirement/candidate p50p95/live-validation separately; missing embeddings and blocked sockets must not affect keyword operation. No nominal scanner-only substitution.
5. Update architecture/user docs and retained QA evidence, run targeted new project-index files and relevant UI governance/CSS checks, scoped lint/format/diff and self-review, then independent spec/quality gate.
ADR required: yes, existing accepted ADR.
ADR path: backlog/decisions/240-workspace-project-context-and-folder-bound-index.md.
Reason: direct implementation of accepted folder-bound storage/tool/management/revocation contracts, alongside ADR101/102/126/150/161/174; no new dependency, semantic implementation, Library policy mutation or navigation API; later steps explicitly authorize cachev3 under ADR240.
6. Retire newly opened registry connections at both writer/supervisor owner boundaries using existing operation_owned_connection injected by runtime. Preserve borrowed ownership and per-call guards; require native-close proof before lifetime/pin/read release, retain failed ownership, and prove actual post-writer/query maintenance drain plus encrypted rollback old-key refusal. This is a concrete K5 recovery gate correction under ADR240/126/228, not authority caching or a global storage change.
7. Implement the accepted bounded multi-directory unit amendment from the keyword plan/ADR240: closed index-only protocol v2, aggregate unit bounds and atomic prepared publication; cache schema v3 indexed attempt revision replaces visited-directory memory. Preserve current admission per unit and final fence, exact job/epoch, identity/completion/pruning/replay, explicit Resume and all native resource limits. Reproduce boundary regressions before code, verify efficient durable frontier, then compare same12-file passive phase baseline before root-approved full literal corpus run. No global admission cache, ordinary executor change, Library mutation or semantic work.

8. Close the production invocation gap: finite registry ownership spans the complete fs_search_index provider entry before all guards/exclusions and complete controller scope preparation, plus direct service/key operations. Reuse existing independently counted native leases and nested borrowed ownership; require outer close before returned evidence and block maintenance through actual close/failure. Prove with real callbacks and no test-only wrapper. Stage actual query native proof, full process retirement, encrypted rollback and fresh old-key refusal honestly; no new authority/token broker.

9. Verify newly owned registry native closure using a small index-specific runtime adapter around existing operation_owned_connection, preserving borrowed identity and generic close behavior. Generic kernel close can swallow failures; actual swallowed/denied close must raise unproven retirement, retain counted ownership and block returned evidence. Reuse this strict scope at new outer provider/controller/UI/service/key boundaries; no new admission or generic helper API.

10. Guard production cache v3 migration before mutation with a runtime-supplied migration-only exact binding lifetime/capacity context: version read outside SQL transaction, nonblocking native lease, whole-binding namespaces/sidecars and DELETE-journal reserve, max_page_count refusal, transactional version recheck/atomic003. Keep v3 status usable and v2 cleanup-only nonmigrating. Prove scaled near-capacity rollback/refusal, successful control/frontier migration, native busy and failed-constructor positive retirement/ownership retention; no generic recovery-owner changes.

11. Fulfill existing spec filename/path keyword behavior in the same v3 FTS path+text schema, including a zero-length chunk for valid eligible files with no text chunks. Migrate/backfill in bounded SQL under the approved migration guard, preserving v2 cleanup-only. Prove filename-only and empty/whitespace matches, filtering before rank/count and fresh Capture omission for changed/deleted files; final v3 candidate/live measurements replace no historical receipts. No new mode or permission.

12. Expose exact admitted nonmigrating management snapshot for legacy upgrade-required Settings/Console and public Clear, preserving scope/CAS and no DDL on visits. Prove actual mounted current-v2 Clear under migration pressure. Move service/runtime initial close lock snapshots offloop inside shielded full drain; held migration heartbeat/cancellation must not orphan or return early.
13. Default to all eligible UTF8 text; preserve valid empty policy tuples as all, remove arbitrary four-extension fallback and blank rejection. Use existing inputs/clear placeholders, mounted saved partial/all policy round-trip and real extensionless/manifest default coverage, with explicit narrowing/security limits intact.

13a. Align worker._allowed with SQL empty=all semantics: no prefixes permits all included folders, no suffixes permits all otherwise eligible UTF8 text; mandatory archive/generated/sensitive/no-follow/size limits remain. Prove actual defaults/partly empty policies, not just UI projection.

14. After immutable source/format completion and physical retirement of both literal corpora, allow the single read-only K5 source review to overlap remaining targeted qualification. Record exact BASE/HEAD; final review verdict and Done still require all final source-qualified receipts. Any later product change receives a scoped diff/review. ADR240 applies; workflow scheduling creates no new architecture decision.

15. Resolve complete K5 review I1: validate draft include/suffix inputs only for Enable/Apply. Disable/Clear/Pause/Resume/Refresh retain saved policy and exact scope/job CAS despite invalid drafts. Prove intended RED then mounted GREEN and full card-file covering. Only card/UI-test files change; old covering UI subset is source-qualified and superseded by post-fix coverage. Core and literal scale sources remain unchanged. ADR240 applies; no new ADR.

16. Resolve K5 review I2 by adapting only the private app-shutdown test factory to supply an honest post-drain retirement proof while delegating existing store behavior. Preserve real held scanner/writer, pipe/thread retirement and complete app shutdown assertions; run failed node plus directly affected cancellation/proof checks. Production _AdmittedStoreFactory proof remains unconditional; no optional fallback/runtime change. Keep original covering failure and corrected focused output distinct. ADR240 applies.

17. Preserve final path+text50+5 common/rare/miss per retained20k cache and20 public Capture samples separately, all actual registry/native retirement. Exact source audit confirms expanded admitted500 failure; retain it explicitly as unmet deviation while original canonical decorated lexical500 remains gate. 18. Fix2 only pre-normalizes current query exclusion predicates in lexical_candidates, preserving all admission/security/SQL semantics and no cross-query cache. Add real-store equivalence/fresh-floor/pre-top-k cases, selected DB/provider covering, exact two-file commit and same-cache final timing. ADR240 applies; no new architecture interface.

19. Final whole-delivery review found I1-I4 and M1-M3; address complete list in ONE consolidated wave. First establish failing held complete-operation close, detached retry, mixed-case real scan/public query, Building partial query and dirty-status cases. Implement finite index operation retirement through outer registry proof, casefold-consistent policy, committed Building eligibility, bounded proven-run history, scanner types and dirty draft preservation. Capture source-backed exact detached cleanup ownership/format/permission/quota/retry proposal and root ADR240 amendment before I2 code; reuse existing native lifetimes/catalog cleanup, no new source authority/global collector. Run only focused amended cases and directly affected covering, scope lint/format/diff and source-timed benchmark applicability; keep prior raw/limits. Commit one consolidated wave and run exactly one scoped rereview. ADR required yes, existing ADR240 amended before cross-boundary changes; semantic and Library mutation excluded.

20. I2 concrete contract: <=4096byte immutable cleanup-owner-v1.lock in exact existing binding group, closed JSONv1 ownership IDs/revision only; actual no-follow/single-link native capability receipt I/O, private parent, quota charge and live-only enrollment/repair before cached contents. No registry fence over native acquisition/flush/bulkcleanup. Detached retry holds current+observed native lifetimes/data pin, validates all catalogs and later tombstone/archived exactowner, serializes same-ID replacement/Enable and refuses unknown legacy ownership. Async detach/archive hints plus Settings Workspaces Retry removed-folder cache cleanup cold-restart entrypoint; complete I1 operation drain. Reuse existing clear_owned_namespace/reclaim/headroom; legacy Clear never requires enrollment allocation/migration. ADR240 existing amended before I2 code; no schema/source authority/global collector.

21. Final scoped review approves with one nonblocking M1 supervisor-tail residual. Explicitly qualify AC12 and retain its source-backed future correction, close the feature with that accepted debt and the measured admitted-latency deviation, retain all reports/raw/visual/ruling evidence and preserve the local branch/worktree. Existing ADR240 applies; no second fix wave, full sweep or integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented optional folder-bound keyword indexing, Off by default, with explicit Enable/Refresh/Pause/Resume/Clear, truthful coverage, Settings/Console controls and durable detached-cache cleanup retry. Saved Library choices/scopes remain intact; no Library copies, semantic indexing, provider or new dependency. Final source3b616994e6ff3d3237a08f2802f25c5b5633e693; service/runtime/store/native lifetime, registry/controller/Console/Settings integration, focused tests and architecture/user docs.

One consolidated final wave and one scoped rereview completed: APPROVE WITH ONE NONBLOCKING MINOR,0Critical0Important1Minor. I1-I4/M2/M3 addressed. M1 history is bounded and native ownership is retained, but a natively retired supervisor may be pruned before its final Python bookkeeping tail exits. This real residual is explicitly deferred under the wave cap; AC12 records the qualification. No native leak/disclosure/corruption or important shutdown defect established; do not claim every physical supervisor exited or all seven findings fully fixed.

Final covering4failed75passed1684.90s preserved; four fixture expectations superseded by12affected/control passes374.84s, including failed-supervisor lock RED/GREEN, normal build retirement and both mounted themes. All four saved Library combinations and native detached/cold/race/receipt proofs passed in covering source. Final same-cache probe1passed233.46s: six canonical lexical p95 targets151-312ms; broader admitted common667/535ms still exceeds500ms, independently accepted explicit deviation. All42Capture/native proofs retired,40public registry returns retired, service0searches/0operations. Original mixed/adverse/setup outputs remain, not an all-green aggregate.

Historical literal20k files across20,200subfolders and separate20k-file single-folder builds remain source-qualified for unchanged worker/publication paths; final lifecycle enrollment adds setup, so build durations are not exact-final-source timings. Windows/headless/native qualification only; staged encrypted recovery, simulated optional-dependency absence and blocked sockets retain their limits. No full suite or hosted/live-provider claim. Final scoped core Ruff/format/diff pass; shared comparison91inherited/0introduced findings; earlier governance/source-attributed failures retained.

ADR required:yes, existing backlog/decisions/240-workspace-project-context-and-folder-bound-index.md amended before ownership/schema changes; ADR101/102/126/174/228 apply. Exact reports,50new raw/source artifacts, original184K5 receipts, current/historical visuals and every chronological ruling/cost are retained in Docs/superpowers/qa/2026-10-20-workspace-project-context/keyword/execution-notes.md and final-re-review.md. Original checkout/user data untouched. Branch/worktree/corpora preserved; no push/PR/merge.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented optional folder-bound keyword indexing with preserved Library choices. Independent final approval:0Critical0Important1acceptedMinor(M1 supervisor bookkeeping tail); canonical query target passes, expanded admitted common667/535ms remains accepted deviation. Exact source3b616994e6, all targeted/source/native/visual evidence and limitations retained in keyword QA execution-notes.md; branch/worktree preserved, no external integration.
<!-- SECTION:FINAL_SUMMARY:END -->
