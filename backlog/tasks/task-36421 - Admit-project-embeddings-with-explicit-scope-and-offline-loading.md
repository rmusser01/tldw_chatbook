---
id: TASK-36421
title: Admit project embeddings with explicit scope and offline loading
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-23 03:27'
updated_date: '2026-10-23 15:44'
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
<!-- SECTION:PLAN:END -->
