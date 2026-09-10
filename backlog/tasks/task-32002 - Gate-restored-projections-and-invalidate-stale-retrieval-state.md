---
id: TASK-32002
title: Gate restored projections and invalidate stale retrieval state
status: In Progress
assignee: []
created_date: 2026-09-07 23:58
updated_date: 2026-09-10 21:42
labels:
- backup-recovery
dependencies:
- task-31978
- task-31989
- task-31990
- task-31991
- task-32000
- task-32001
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: No stale or incompatible projection can serve restored source data, including after relaunch or rollback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No stale or incompatible projection can serve restored source data, including after relaunch or rollback.
- [ ] #2 Omitted/shared indexes obey explicit previewed retirement/quarantine and scope rules.
- [ ] #3 Retrieval resumes only after qualified compatibility and reconciliation, without automatic rebuilds.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Original component04 Task19 only. Start with pure RAG definitions/persistent-root discovery and correct unused/empty states using exact installed selectors; add actual indexing SQLite schema adapter and qualify finite definition writes through existing admission. Then establish supported persistent engine capture lifetimes, and implement original durable projection quarantine/readiness plus actual query/cache generation gates when restore plan/journal dependencies exist. No engine startup during discovery or automatic rebuild. Source map: /private/tmp/chatbook-rag-task19-source-map.md. Preserve unsupported nonempty projection coverage until actual qualification; no blanket placeholder removal.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started original Task19 discovery/capture sub-slice because missing rag.definitions/rag.projections currently block all complete inventory even with absent RAG stores. Restore quarantine/readiness still depends on original Tasks17/18 and is not claimed complete.
Inert factory and exact RAG indexing SQLite adapter implemented without importing RAG runtime or vector engines; original RAG package startup remains unchanged. Existing core borrower/transaction/close seams, literal v0 schema, source-preserving native snapshot and exact main-thread idle close verified. Definition/projection files remain explicitly unsupported until actual writer/engine qualification. Focused existing/new25 passed, latest discovery8/indexing6 passed; review caught and fixed auto env fallback. Root public factory integration then exposed semantic misuse of shared_group; removing those labels rather than weakening physical identity checks. Remaining Task19 runtime/projection readiness contracts are not complete. Source report /private/tmp/chatbook-rag-discovery-report.md.
Installed inert RAG factory integrated with public capture. Removed semantic shared_group tags; that field retains physical alias identity only. Actual public discover/classify regression covers absent stores and two retained definition files. Three public capture variants and final combined146-test cohort pass; RAG production Bandit0. Nonempty RAG definitions/Chroma and restored retrieval gates remain explicitly unqualified; Task19 stays In Progress.
Bounded RAG profile native admission implemented in config_profiles.py. Exact synchronous manager/root scopes cover mkdir, CRUD, loader self-heal, legacy migration and selected experiment output paths, preserving existing mutation/error semantics. Root rerun10 focused passed52.55s, including12 unchanged existing profile test functions in fixed-selector child. Direct existing44-test pytest cohort encountered documented root-fixture selector drift (42fail2pass); ineffective adaptation removed, source guard preserved. Scoped review no findings, report /private/tmp/chatbook-rag-profile-admission-review.md; implementation/source map /private/tmp/chatbook-rag-profile-admission-report.md. Ruff38→38/Bandit2→2, no new findings. No runtime/engine/pipeline/format or pending-marker qualification claimed; remaining original Task19 work is explicit.
Original Task19 durable-definition prerequisite: first-use pipeline copies and pipeline export now retain ordinary storage admission through existing synchronous mkdir/copy/open/dump completion. Builder cache initialization follows accepted source selection so refused copies remain retryable. No format qualification or capability-marker removal. Agent observed 9 behavioral reds before patch; root independently ran test_rag_pipeline_admission.py + test_pipeline_middleware_contract.py: 23 passed in11.02s (/private/tmp/chatbook-root-rag-pipeline-green.log). Agent broader focused cohort96passed/4known raw_source_selection_changed Console fixture failures; no fixture changes. Production Ruff107→107; Bandit0→0; exact6 write census rows updated separately from read-open rows. Report /private/tmp/chatbook-rag-pipeline-admission-report.md. ADR-126; Task19 remains In Progress for runtime and restored-projection integration.
Retained RAG credential-format sub-slice implemented in credentials.py + new test_rag_definition_credentials.py: actual legacy blobs/.migrated and shipped pipeline TOML use existing semantic sanitation/encrypted material; validated experiment config/results remain byte-exact historical content. Unknown paths/shapes refuse coverage; unreadable rollback retains bytes with issue. Real ProfileConfig optional URL null now preserved. Behavioral15red+5controls -> new45pass7.62s; affected new36+existing60=96pass18.41s. Ruff/format/diffclean; productionBandit0->0, tests30LOWassertions only. Report /private/tmp/chatbook-rag-definition-credentials-report.md. Existing start_experiment Path JSON serialization issue documented, not fixed; no RAG definition lifetime/runtime/activation qualification claimed. Source frozen for review, not committed.

Bounded native lifetime prerequisite authorized by root: close the three finite collection_indexes clients on every exit using the public installed Chroma close API. Declared dependency chromadb>=0.4.0 requires preserving older API behavior while explicitly warning unqualified retirement when close is unavailable/fails. No resident vector lifecycle or capture-ready claim. TDD real private fixed-selector subprocess tests with network guard and explicit embeddings; scoped existing tests, Ruff/Bandit baseline. Own collection_indexes.py and new test_chroma_collection_lifetimes.py only; no commits. MCP reads pending/unresponsive, official CLI fallback used before edits.

Finite collection client lifetime slice ready for review: three source-local finally cleanup calls use supported public close; missing/failed close explicitly logs chroma_retirement_unqualified while preserving ordinary >=0.4.0 API behavior. New tests red15 then final15passed11.04s with real private Chroma1.5.8/refcount/shared-engine/reopen vectors and guarded fixed selectors. Existing focused17passed2known raw_source_selection_changed fixture failures; no weakening. Ruff4to4 (new deliberate cleanup BLE001 rationale), newtests0; Bandit0to0. Report /private/tmp/chatbook-chroma-collection-lifetimes-report.md. No capture qualification, marker removal, staging, commits or TaskDone.

Review correction: replace close logger.exception with fixed sanitized warning. Syntheticsecret close-error red3 then targeted compatibilitygreen6; no exceptioncontents in capturedlogs. Exact collection_indexes diagnostic inventory rowonly updated via existing scan_source/digest API:8calls5c9579976f10a76c75f4, TASK494 stillpending. Scoped literal/noexception guard passes. Globalguard1failure due20otherownerrows and existingtopology/aggregate drift, detailed /private/tmp/chroma-diagnostic-drift.json; no blanketupdate. Report appended; nostaging/commit.
Root final review of finite collection client cleanup: independent test_chroma_collection_lifetimes.py run 15 passed in13.21s including sanitized close errors (/private/tmp/chatbook-chroma-root-final.log); scoped production Bandit0, new tests Ruffclean, diffcheckclean. Accepted narrow public-close finally behavior and exact pending diagnostic owner row; source missing-close/failed-close remains explicitly unqualified. No resident/projection activation completion claim. Preparing scoped commit only.
## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-19)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

Authorized next bounded prerequisite before edits: source-local finite RAG definition producer settlement for exact delete_user_profile, ensure_imported_profile, SettingsScreen._persist_library_rag_save sequences, composed before global storage pause. Read-only observable active experiment and alternate manager/results source refusals; no duplicate native write admission, Chroma/model/serialization repair, or capture-marker removal. Small private TDD fixtures and scoped static/census checks; no staging or commits.

RAG definition producer prerequisite ready for review: source-local synchronous cohort wraps exact delete/import/Settings profile+config sequences; RuntimeMaintenance closes/drains before draftprobe/globalpause. Read-only activeexperiment/alternateknownmanager/results/unknowninjection refusal preservesbytes. Behavioral3sequence reds then15new+runtime pass11.84s; finalaffectedprobe5pass7.11s. ExistingUI/config7pass1fail86errors allknown raw_source_selection_changed fixture boundary, unchanged. Ruff110to110/newmodule-test0; Bandit1to1/newmodule0; no new diagnostic/nativewriter census rows. Report /private/tmp/chatbook-rag-definition-settlement-report.md. No Chroma, markerremoval, staging, commits or TaskDone.
Root review of finite definition settlement: no actionable findings within explicit3source-method acceptance boundary. Independent test_rag_definition_settlement.py + test_runtime_maintenance.py15passed12.72s (/private/tmp/chatbook-rag-settlement-root.log), preserving underlyingprofile/config outcomes and nativeacceptedwork throughcancellation; no extra native/storagewrapper. Source unsavedprobe onlyloadedexactmanager/service references and boundconfigcache; activeexperiment/unknown/alternate sources refuse without mutations. Agent originalscopeRuff110->110/Bandit1->1, newmodule/testclean; no diagnostic/census additions. Ready scopedcommit. EarlierqueuedUI action enrollment and unobservableuncachedmanager finalcoverage remain explicit prerequisites beforeComplete; no definition/projection marker removed.

Bounded follow-up approved: exact ConfigProfileManager weak instance census and admitted constructor; source-local experiment transition coordination rejects maintenance before sealing active experiments; protect EnhancedRAGServiceV2 start/end before memory changes. TDD focused private tests, no queued Settings edits or adapter marker removal.
Approved bounded queued Settings RAG definition settlement design (proposal section1), before edits: own narrow settings_screen dispatch/native/UI-result integration plus one source-local queued-action helper and focused tests. Reserve DefinitionParticipant explicit token before dispatch state mutation; exact installed Textual Worker._task completion only arbitrates cancellation-before-native-entry, never infers running native completion. Native+call_from_thread delivery retain token; immediate save/activation descendants reserve child explicitly while root intake closes, modal decisions release before user waits and later confirmation uses fresh admission. Preserve exclusive groups; enqueue errors retire reservation before mutation. Coordinate token API with capture_gap, no edits to participant/config_profiles/runtime/capture refusal/activation. TDD actual Textual before/after-entry cancellation and private fixed-selector Settings chain proof, scoped Ruff/Bandit/census only.

Instance/experiment prerequisite implemented (not complete capture): ConfigProfileManager weak census + admitted constructor; experiment transition count and loaded-instance active experiment check refuse maintenance before sealing; exact EnhancedRAGServiceV2 start/end protect service memory. Six new private native-source tests red, then 17 manager/definition tests passed (18.69s), runtime 4 passed (1.50s). Ruff 83->83 same code counts, Bandit 2->2 existing; zero new profile-owned source/diagnostic census rows. No queued Settings/token changes, no marker removal or engine construction. Report /private/tmp/chatbook-rag-manager-maintenance-report.md. Source frozen for root review.
Manager/experiment prerequisite independent review approved for spec compliance and quality, no actionable findings: /private/tmp/chatbook-rag-manager-maintenance-review.md. Final source63lines plus6new private childcases;17new/existingdefinition and4runtimecases passed. Newtestformat corrected, participant/testformatcheckclean and diffclean; existing Ruff83/Bandit2 unchanged, no new producer/diagnostic census rows. Root reviewed locking, weak lifetimes and wrapperentry, preserving activeexperiment metric recording after refusedmaintenance. OriginalTask19 remains InProgress; queuedUI continuation settlement, nonemptypubliccapture qualification and restoredprojectionreadiness still incomplete.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->