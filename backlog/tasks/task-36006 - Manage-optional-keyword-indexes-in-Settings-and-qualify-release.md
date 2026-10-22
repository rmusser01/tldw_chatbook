---
id: TASK-36006
title: Manage optional keyword indexes in Settings and qualify release
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 14:32'
updated_date: '2026-10-22 01:25'
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented optional keyword-index management in Settings and Console with explicit Enable/Refresh/Resume, exact folder/scope/job CAS and stale-view fences. Defaults preserve blank=all and saved Library choices/scopes; no Library ingestion or semantic implementation. Added bounded multi-directory wirev2, durable cachev3 frontier/path+text FTS, capacity/native-lifetime guarded migration, nonmigrating legacy Clear and strict complete-invocation registry retirement. Local query exclusion normalization preserves all production authority checks.

Both literal private corpora completed:20,000files across20,200subfolders and20,000files in one folder; real marker/Capture/native/cache/registry retirement proven. Actual app journeys and staged encrypted rollback/fresh old-key refusal qualified. Final canonical p95 common/rare/miss168-233ms,50samples after5warmups per cache/workload. Expanded admitted common534/562ms still misses later root-added500ms extension: explicit accepted requirement deviation, not a passing metric. Instrumented whole-tool p95~2.70/1.96s; retain all prior failed timing/setup receipts.

Original eight-file cover1failed181passed3skipped1deselected; local factory proof fixture corrected with exact5testpass, full post-fix card23pass, omitted real Windows junction separatelypass; fix2 six DB/provider nodespass and final timing1pass. No full suite. Scoped lint zero introduced,99shared baseline findings; governance14fail attributed by unchanged-source comparison, not exact-base runtime reproduction or green run. Windows/headless/blocked-network and staged-recovery limits preserved. Both independent task verdicts PASS with accepted deviation/qualifications,0newfindings; scanner public typing Minor goes to final whole-branch review.

Files: project-index service/runtime/store/scanner/protocol/worker, provider/controller ownership integration, Settings binding card/Console status, focused tests, architecture/user docs. See Docs/superpowers/qa/2026-10-20-workspace-project-context/keyword/execution-notes.md for exact reports/raw184receipts/visual hashes/chronological rulings. ADR required: yes, existing backlog/decisions/240-workspace-project-context-and-folder-bound-index.md, amended before schema/native ownership changes; ADR101/102/126/174/228 apply. No new dependency or generic authority cache. Whole-delivery final review remains separate and pending; no push/merge.
<!-- SECTION:NOTES:END -->
