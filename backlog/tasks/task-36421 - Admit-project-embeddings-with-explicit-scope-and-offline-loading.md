---
id: TASK-36421
title: Admit project embeddings with explicit scope and offline loading
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-23 03:27'
updated_date: '2026-10-23 17:08'
labels: []
dependencies:
  - TASK-36420
references:
  - backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
  - Docs/superpowers/plans/2026-10-20-workspace-project-semantic-index.md
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow project embeddings to use their own admitted folder lifetime while preserving Library defaults and honest optional-feature availability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Actual embedding constructor, activation and native worker entrances select the exact code-issued project scope before effects, refuse forged/stale/wrong-target or dead transfers, and never infer Library source owners.
- [ ] #2 Accepted async/native work retains the actual live parent and physical retirement; default Library scope behavior remains compatible.
- [ ] #3 Project local embedding loads use a complete existing local snapshot with offline-only loading; a missing model or optional dependency reports setup required without download or keyword failure.
- [ ] #4 Actual embedding factory success/retry/failure and propagated errors expose no raw credential-bearing endpoint or request secrets; only secret-free configuration/destination references are persisted.
- [ ] #5 Model, configuration, artifact and destination compatibility distinguishes same-dimension changes and never treats Library policy as project consent.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read Task36421, the focused brief and final Task1 producer handoff, actual RAG/model-artifact/native admission architecture and existing wrapper/factory tests. Consume reviewed versioned refs/settings and finite SQL/native contracts, preserving Library defaults.
2. Write meaningful REDs through installed constructor/source admission, sync/async/native entrances and actual factory for exact issued project scope, forged/copied/stale/wrong-target/dead-parent refusal, retained accepted-parent transfer and repeated cancellation retirement. Keep default Library compatibility and missing-embeddings keyword controls.
3. Implement ProjectionExecutionScope and the narrow project issuer/resolver, routing explicit project scope before Library source inference and constructor effects. Resolve real existing model catalog/provider pairing and current credentials privately; use versioned catalog refs and SHA256-v1 fingerprints over safe canonical fields, preserving destination consent and distinguishing artifact/model/config changes of identical dimension.
4. Add explicit local_files_only propagation to actual wrapper/factory for project local models, complete existing snapshot validation and offline-only ordinary/restored loading without dynamic fallback/download. Keep optional-dependency and recovery/artifact/native borrower gates, no unused Chroma, no decorator unwrap or Library impersonation.
5. Exercise actual factory success/retry/failure and propagated errors with controlled loopback HTTP/synthetic secrets, safe diagnostic/metric/circuit-breaker identities; no external project text. Qualify production cached local factory offline in isolated qualification runtime if usable and record missing/setup conditions honestly. No install/shared environment mutation or quality claim from deterministic vectors.
6. Run only new/directly affected scope/factory/offline/privacy/activation/native and Library-default targets, scoped Ruff/format/source guards and diff. Retain exact raw/child receipts and adverse outcomes; report final interfaces, accepted-parent reuse/close contract, native retirement and limits. Commit exact task paths for independent review before generation/search work.
ADR required: yes, existing amended decision.
ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md.
Reason: direct implementation of accepted folder-bound semantic projection and destination-consent boundary; no new persistence owner or Library policy change.

Demonstrated Windows offline-loader correction: actual qualified factory reached model_recovery._closure and failed because that owner imports stdlib os while pinned_directory returns the existing platform filesystem descriptor. Python Windows stdlib os.listdir(fd) is unsupported. Reuse tldw_chatbook.Utils.platform_files.os in model_recovery.py, as native_files already does, so listdir/open/fstat use the existing handle-relative no-follow Windows implementation and unchanged stdlib POSIX behavior. This is a narrow import/adapter correction under existing ADR126 and ADR241, not a new storage/source permission or Windows backend. Do not add path-based enumeration fallback, weaken alias/single-link/identity/manifest/closure limits, change the generic platform shim or bypass model/recovery gates. Retain the actual offline13/16/17 failures and run native closure success/refusal tests plus directly affected model-reference/recovery controls and actual blocked-network offline factory. Report any additional owner need before edits.
ADR path also applies: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: direct reuse of existing Windows-native filesystem contract; no new ADR.

### Task2 review correction: retained project model close failure

Ruling: Use only the existing ProjectIndexService operation owner to strongly retain failed project model retirement. Add a private projection_close_pending token-to-actual-wrapper map to _Operation and narrow private mark/settle/check helpers under the service lock. Register the actual project wrapper before factory/model close; keep its actual factory/cache record and carrier/retained parent on any failure, raise/log only classified project_embedding_close_failed without raw cause, and remove only its token after positively successful retry. Carrier close and wrapper use/rebind refuse while its retirement is pending; the service-rooted operation -> wrapper -> carrier -> actual parent chain must survive caller reference loss and GC. A self-contained wrapper/carrier cycle alone is insufficient. Default Library close behavior is unchanged.

Keep projection-close pending separate from generic retirement_unproven. It must not blanket-deny registry_connection_scope or keyword reads merely because a model close failed; new semantic use on the affected binding refuses, while actual keyword lookup remains usable. During explicit service shutdown, detect/report a known unresolved projection close before waiting forever on its not-done operation; retain its real owner and permit positive same-wrapper cleanup retry. Never clear an unrelated native/store/capability uncertainty, claim cleanup success, or release the parent because an exception was swallowed. Serialize close/rebind and prevent embed reuse after partial closure. No SQLite handle/gate/fence spans model close, joins or waits.

Authorized additional exact product path is tldw_chatbook/Workspaces/project_index/service.py, limited to this existing operation-owner retention/reporting seam. Existing Task2 projection_scope.py, embeddings_wrapper.py and actual factory paths may consume it; no generic recovery framework, new persistence owner, source grant or Library behavior change. Test actual factory record retention after injected close failure, sanitized logs/errors, caller-reference loss/GC, carrier and affected-semantic admission refusal, actual keyword lookup availability, explicit shutdown unresolved result, and successful physical retry followed by parent retirement. Root inspected actual factory close (cache.clear only after all record closes), service operation roots/admission and shutdown waits before this amendment.

ADR required: yes, amend existing ADR241 before the cross-module owner change. ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md. Reason: narrow completion of accepted physical retirement and keyword independence, reusing existing project operation owner; no duplicate ADR.
<!-- SECTION:PLAN:END -->
