---
id: TASK-32671
title: Commit and recover reviewed plugin installation changes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:18'
updated_date: '2026-09-16 08:24'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32670
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make reviewed local installation and authority changes recoverable without accidentally publishing uncommitted capabilities.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review binds exact package/catalog digests, installation identity, interpretation, selection, workspace and authority inputs; stale reviews cannot commit.
- [x] #2 Publication follows prepared snapshot and intent, durable SQLite commit, post-commit certificate, secure marker, then projections; retries use the same operation identity.
- [x] #3 Crash recovery follows every specified evidence state in another process, preserving unrelated installations and refusing automatic promotion when commitment proof is missing.
- [x] #4 Full disks, missing responses and interrupted writes preserve old or fenced committed state without recreating grants or executing package code; recovery material cannot be pruned.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md; extends backlog/decisions/009-local-skill-trust-boundary.md
Reason: Direct implementation of accepted exact review, owned durable commit/certificate/marker ordering and complete authenticated recovery; no new permission or runtime owner.
1. Read the reviewed F1-F3 interfaces and task brief, relevant lessons and ADR contracts; create real plugin_stack fixtures and establish RED through coordinator/recovery entry points using the isolated interpreter/profile.
2. Implement frozen exact PluginReview and stable operation receipts binding retained package identity, interpretation, selection/dependencies, workspace/configuration/mappings and current authority. Revalidate immediately before mutation; preserve disabled/untrusted installation defaults and operation identity across retries.
3. Implement the six-step owned commit path: reconcile incomplete transitions, stage immutable package material and protected complete snapshot/intent, durably commit SQLite, create certificate after transaction return, advance exact marker, then publish. Keep affected authority fenced and serialize SQLite creation/use/close on one worker without blocking the UI thread.
4. Implement authenticated recovery and exact receipt lookup. Under R19 minimally extend Plugins/authority_store.py with bounded public transition discovery and Plugins/registry.py with needed recovery/projection APIs, plus their focused tests. Verify exact identities/MACs before the recovery classifier; preserve unrelated installations, tombstones/root fences and retained material, without external grants or inferred process cleanup.
5. Use real independent recovery_worker processes and named post-durability barriers to kill the owner after each boundary, then recover with fresh process state. Cover prepared-only, old-marker/certificate, committed DB without certificate, advanced marker with missing/rolled-back DB, invalid evidence, stale review, full disk, lost response and repeated operation ID. Missing runtime provenance retains a conservative recovery gate.
6. Consume F3 bootstrap/reset only under the existing F2 owner; reset requires retained operation_id (R16). Reconstruction reinspects immutable bytes and exact adapter/definition references (R12); unknown/missing overlays/configuration remain Recovery required. No fetch, package execution or blanket PID termination during recovery.
7. Run exact feature and changed-seam regression files, required static checks and docs updates; self-review and write task-4-report.md. Controller owns exact-path commits, independent task review and AC/Done closeout. No full-suite run, shared environment mutation, real profile/keyring or separate runtime. F8 owns later durable root-user relationships; M4 owns later MCP request/connection ownership.
8. R20: coordinator/registry stay on one dedicated non-UI worker and persistent event loop; refuse main/UI-thread mutation or wrong-thread use. No transparent handle transfer or per-call asyncio.run. Real worker fixtures exercise this contract; lazy application worker/marshalling is integrated with the existing app owner in the native skill task.
9. R21: after provenance loss, record idempotent unresolved process evidence with reserved unknown prior-owner identity and no invented PID through minimal registry helpers. Preserve real surviving rows; prove fresh-owner reserve_launch blocks affected installations. This is a missing-evidence gate, never reconstructed process liveness or cleanup proof.
10. R22: initial public review/commit serves new local installs with stable fresh installation identity and explicit current authority/config/mapping absence or values. Exact retries preserve operation identity; existing-ID overwrites/arbitrary authority JSON endpoints are refused. The common authenticated pipeline is extended by explicit trust/activation and later drain-bound revision reviews in their assigned increments; catalog binding arrives with marketplace sources. Embedded package overlays remain eligible for exact reinspection, while missing external catalog artifacts block recovery.
11. R23: bound retained authenticated transition discovery to1,000, refusing a new prepare at capacity while preserving exact retry/recovery. Do not prune in this task; rolling eligible terminal transition/receipt retention is integrated with protected material in the assigned revision retention task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented exact reviewed local installation and crash-safe authority publication through the existing plugin owner. Session-bound reviews cover package bytes, source/link provenance, interpretation, selection, workspace and current authenticated inputs; installs start disabled and untrusted. The guarded SQLite transaction prepares the complete protected snapshot before commit, then writes a separate certificate, advances the marker and publishes. Recovery validates exact marker ancestry and reconstructs all retained authority from verified material, preserving other installations and persistent unresolved process gates.

Changes: Plugins/coordinator.py, recovery.py and review.py; narrow registry/store helpers; real persistent-worker and killed-owner tests; Plugins/README.md. Dedicated worker integration, typed activation/revision operations, mapping-owner validation and rolling retention continue in their assigned increments. No separate runtime/permission owner or schema migration was introduced.

Validation: 296 targeted affected/neighbor tests passed in28.46s before final local self-review fixes;44 final feature tests passed in18.10s. Independent review found one ancestry gap; fixed with RED4failed/4passed, GREEN8passed and51 covering tests passed in19.75s. Counts overlap. Ruff, Black(py312), syntax and whitespace checks clean. Fresh scoped review confirms I1 addressed and no new issues. No full sweep.

Qualification: real controlled SIGKILL at all six durable milestones and fresh-process recovery on local macOS/APFS with isolated file marker backend. Actual keychain, hardware power loss, other platforms and filesystems are not qualified. Children verified worktree import and configured scratch data roots; no production profile/keyring, network/package execution, inferred external grants or arbitrary PID signals.

ADR required: yes. Implements backlog/decisions/162-managed-agent-plugins.md and163-expanded-console-hook-runtime.md, extending009-local-skill-trust-boundary.md. R19-R23 record bounded discovery, dedicated worker, conservative missing-process evidence, staged typed operation scope and capacity/retention handoff. No new ADR needed for the ancestry bug fix. Implementation commits a993777d7d and2c1a618233. Relevant existing isolation/evidence lessons applied; no new generalized lesson.
<!-- SECTION:NOTES:END -->
