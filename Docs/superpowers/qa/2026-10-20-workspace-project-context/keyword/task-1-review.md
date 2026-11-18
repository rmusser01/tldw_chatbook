# K1 task review

## Spec Compliance

**Verdict: Spec compliant for the K1 foundation, with the later-task integration gates below.** No blocking missing, extra, or misunderstood K1 implementation was identified against the brief and its recorded clarifications. This is not approval of the complete keyword-index feature or a merge verdict.

The package contains every requested production/test change: immutable shared contracts, schema-v1 cache, schema-v9 Workspace authority migration, registry reader/fence, exact owner/closer registration, excluded recovery declaration, and targeted migration/ownership/store tests. The additional prepared-page seam and inventory update are explicitly authorized by the brief. Root-owned task/ADR metadata was intentionally excluded from the implementation commit.

## Strengths

- `tldw_chatbook/DB/Workspace_Project_Index_DB.py:52` prepares chunks outside publication, with explicit 400-word/100-overlap arguments and source line offsets. `:450` rejects raw pages rather than doing expensive implicit preparation under the caller's fence.
- `tldw_chatbook/DB/Workspace_Project_Index_DB.py:288`, `:361`, and `:479` enforce durable claim/revision/owner checks, exact orphan-claim replacement, duplicate-page receipts, and transactional manifest/text/FTS/frontier publication. Real connection/process tests cover competing admission (`Tests/DB/test_workspace_project_index_db.py:119`, `:207`, `:288`), and injected failure covers publication rollback (`:90`).
- `tldw_chatbook/DB/Workspace_DB.py:28` installs non-cascading authority witnesses; `:723` migrates through the existing transaction runner. The frozen recovery successor is added alongside v8 (`tldw_chatbook/DB/recovery_operations.py:142`). Rollback and catalog compatibility have direct tests (`Tests/DB/test_workspace_db.py:906`; `Tests/Backup_Recovery/test_workspace_project_index_owner.py:68`).
- `tldw_chatbook/Workspaces/registry_service.py:2649` checks the exact key, data authority, local binding, locator and ancestor identity; `:2713` uses an explicit immediate registry transaction. The process contention test at `Tests/Workspaces/test_workspace_registry_service.py:1257` tests the actual serialization boundary.
- `tldw_chatbook/DB/Workspace_Project_Index_DB.py:643` publishes Off before cleanup, and `:669` refuses stale cleanup after re-enable. The observer test at `Tests/DB/test_workspace_project_index_db.py:355` proves both visibility and retained CAS history. `:392` tests narrowed processing policy against retained candidates.
- `tldw_chatbook/Workspaces/project_index/recovery.py:9` classifies detached derived files as excluded and rejects capture. Owner tests at `Tests/Backup_Recovery/test_workspace_project_index_owner.py:50`, `:85`, `:99`, and `:111` exercise native-handle retirement, activation denial, import inertness, and maintenance settlement rather than only checking registry strings.

## Issues

### Critical / Important

None identified within K1's agreed scope.

### Minor

1. **Public adapter return contract is unannotated.** `tldw_chatbook/Workspaces/registry_service.py:2649` adds the cross-task public adapter factory without a return annotation, although the exact `IndexAuthorityReader` and `IndexAuthorityFence` aliases already exist. Annotate the returned pair (and preferably the public fence's yielded row type) so later composition/static checks can validate the intended seam. This is maintainability polish, not a functional blocker.
2. **Verification output remains noisy (inherited).** `.superpowers/sdd/2026-10-20-workspace-project-keyword-index/k1-final-focused.log:1` contains the unset pytest-asyncio fixture-loop-scope deprecation; `:8` contains Pydantic's deprecated Field metadata warning. The report correctly discloses these and the inherited lint findings (`task-1-report.md:117`). Track/fix them with their existing owners; do not suppress them or broaden K1 to unrelated cleanup.

## Cannot verify from this diff / later integration gates

- **Real profile/restore authority issuance:** `registry_service.py:2649` deliberately takes a code-owned callback. K3 must qualify it against actual admitted profile/store identity and restore/activation generation. Synthetic IDs in K1 tests establish comparison behavior, not production issuance.
- **Physical worker retirement and safe management orchestration:** `Workspace_Project_Index_DB.py:361` verifies supplied retirement evidence and performs exact CAS; `:416` documents retirement as a caller obligation. K3 must prove actual contained-worker/process retirement before releasing/replacing claims, including configure/revoke followed by a new Enable/Resume. It must pass expected scope for management actions and keep cleanup outside the registry fence. A store claim deletion alone is not retirement evidence.
- **Publication/disclosure serialization in production:** the registry fence exists, but no K3/K4 production caller is included yet. Later tests must show registry-before-cache ordering, current authority/exclusion checks and no joins/scans/cleanup under that fence. `Workspace_Project_Index_DB.py:583` returns metadata candidates; it cannot itself prove the caller held the registry fence.
- **Live-text and output safety:** `models.py:273` onward supplies candidate/live-reader/search-authority contracts. K4 must perform same-read live identity/hash validation and enforce at most 10 results/16 KiB (and smaller run budgets). The internal candidate limit permits up to 100 at `Workspace_Project_Index_DB.py:583`; that is not a user-facing result cap or authorization to return 100 hits.
- **Scanner/resource guarantees:** K2/K3 must enforce the two-read concurrency bound, aggregate 4 MiB writer queue, eligible text/generated/binary/reparse exclusions before reads/descent, explicit refresh/resume and complete traversal/reconciliation. K1 bounds individual pages/captures and its current DB/sidecars; it does not demonstrate a production scheduler or whole binding storage accounting across obsolete namespaces/future staging/vector files.
- **Delivery/UI/Library behavior:** no new production scanner, tool/provider, UI, watcher or Library integration is in this task. Saved Library controls and absence of Media/Notes writes are consistent with the diff, but end-to-end production behavior remains a later gate.
- **Workflow metadata:** the report states root owns Backlog/plan/ADR execution records; their final status/evidence is not established by this commit package.

## Review checks and evidence

Read AGENTS.md, the supplied reviewer template, task brief with clarifications, implementation report, and the complete supplied diff. The diff was read in bounded sequential portions; one tool response truncated the store's lexical-result/revoke/cleanup boundary, so that omitted portion was recovered from the diff package. Changed source files were not separately crawled, and no git command, suite rerun, install, profile mutation, or product edit was performed.

Named unchanged-source interface checks:

- **Exclusion semantic drift:** inspected `Utils/sensitive_paths.py:1007`'s SensitiveExclusion contract. The new matcher implements subtree, exact-file, arbitrary-depth filename and direct-child-file behavior for file candidates consistently with that contract.
- **Chunk offset/parameter contract:** inspected `RAG_Search/chunking_service.py:128` and its adjacent flat-contract adapter. The called API accepts explicit size/overlap and documents the character-offset fields consumed by preparation. No competing chunker or embedding seam was introduced.
- **Transaction/journal capacity assumptions:** inspected `DB/store_kernel.py:120` and `:175`. The kernel sets true autocommit and exposes the existing immediate transaction option without introducing a different journal policy; K1's explicit registry immediate fence and cache transaction use match that interface. This source check does not certify future additional cache files against the physical cap.

Inspected retained log tails rather than regenerating evidence: `k1-final-focused.log:34` records 25 passed, 1 warning; `k1-baseline-census.log` ends with the two inherited inventory failures described in the report. The covering run remains 4 failed/183 passed/10 skipped as reported, with the two K1-owned failed nodes subsequently included in the final passing selection. No claim of a wholly green broad covering run is made.

## Assessment

**Task quality: Approved with minor notes.** The foundation has explicit contracts, narrow ownership, transactional state changes and meaningful native SQLite/process tests. Approval is limited to K1; the named authority/lifecycle/disclosure/scanner integration gates must be qualified by their owning later tasks.
