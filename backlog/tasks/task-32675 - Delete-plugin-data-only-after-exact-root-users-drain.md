---
id: TASK-32675
title: Delete plugin data only after exact-root users drain
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 04:21'
updated_date: '2026-09-16 20:48'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32673
  - TASK-32674
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove only the reviewed saved data after every owned user has stopped, including processes that can write while idle.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Deletion review binds installation identity, exact owned roots, generations and all affected workspaces; a persisted access fence precedes destructive deletion.
- [ ] #2 Shared-root users and idle writer-capable processes are tracked until confirmed stopped; unresolved writers or handles leave deletion pending without forced cross-workspace cancellation.
- [ ] #3 Reattachment, stale root generations, symlink replacement and PID reuse cannot redirect cleanup; partial cleanup/restart retains the fence and honest progress.
- [ ] #4 Successful deletion advances root generation before fresh authorized use; cancellation cannot claim data restoration or revive cancelled work.
- [ ] #5 Qualified clean shutdown permits safe root recovery after restart; dirty, unknown, malformed or rollback-lost ownership evidence keeps affected retained roots fenced until explicit reconciliation, without treating an empty restored registry as proof that users stopped.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: Implements the accepted saved-data fence/drain/cleanup contract and records R18/R43 exact-root persistence plus the separate protected clean/dirty session checkpoint. Amend these existing decisions and managed-plugin spec before code; no new runtime, cleanup journal or permission owner.

1. Read task-8-brief.md, actual TASK-32675, final reviewed F7 report/APIs and f8-root-ruling.md (controlling), with f8-root-ownership-proposal.md and f8-owner-proof-research.md as evidence. Recheck final F7 schema/migrations and use plugin migration 004_root_users.sql (current committed F7 schema is 3; recheck HEAD after the review gate). Confirm retained pin, issued-ID, review/result and terminal-owner contracts; notify controller before extra paths.
2. Amend the existing ADR/spec with immutable original owner/current attachment, root-generation/binding/coverage and conservative deletion phases, plus R43 protected checkpoint ordering and reduced-backend limits. Preserve historical absent fields/canonical bytes. Record actual macOS/APFS limits, not SDK-only qualification.
3. Establish RED through the real coordinator delete_data entry after successful reviewed root establishment and exact controlled child registration. Observe that an idle writer still blocks while revision leases are zero; stop/reap the exact child and use the same entry as the successful control. Isolate profile/interpreter/network/keyring and all child resources.
4. Add data_cleanup.py and the next root-user migration, minimal registry/runtime-owner APIs, closed root bindings and expected-grants commitment. Persist pending owner plus complete joins before grant; keep live pending claims across worker IO, recheck epochs before publication, and settle only from actual host terminal evidence. Old-generation, idle, pending, read and shared-workspace users remain blockers; missing evidence never means zero users.
5. Extend the existing marker backend and service lifecycle with R43 separate scoped runtime checkpoint, bounded closed encoding and write/readback checks. Dirty precedes any root grant or root-owning operation; clean follows admission closure, complete confirmed drain and registry settlement, before owner release. Preserve PluginMarker/F4/R37/F7 identities and immediate F6 sealing. New-session dirty/unknown/mismatch fences all retained roots, including orphans. Initial bootstrap, legacy reconciliation, failed publication and reset-with-retained-data require explicit tests.
6. Implement root creation/binding, detach/reattach and exact deletion review through the same coordinator pipeline. Freeze exact reviewed roots/physical bindings/affected scope; default deletion waits and explicit cancel-work remains separate. Each durable phase has its own issued ID and request nonce; grouping metadata is not an ambiguous result alias. Read-only lookup never starts destruction.
7. Commit the fence before deletion and a deleting phase before first unlink. Revalidate actual no-follow descriptor-relative root/anchor, generation/attachment/fences before each destructive operation. Use a qualified exact physical identity, accounting for the observed Python metadata precision gap. Reject replacements/unknown identity; preserve partial cleanup/fences, then advance generation only after confirmed removal and durability. Cancellation clears only its own reconciled pre-deletion fence and never restores old work.
8. Update final F7 admission/pin data coverage with attachment, exact membership, original/current ownership, scope, binding, generation and fences. Preserve package-only qualified-none as explicit producer proof. Integrate existing Console/fleet host terminal callbacks only if necessary; M4 later owns actual MCP connection/request grants and lifetime qualification.
9. Source scope: new Plugins/data_cleanup.py and Plugins/migrations/004_root_users.sql; existing Plugins/runtime_owner.py, registry.py, authority.py, authority_store.py, coordinator.py, recovery.py, admission.py, service.py, review.py and continuation.py. No previous migration edits or main Chat DB schema. Additional real-owner paths require controller scope update before code.
10. Run exact new Tests/Plugins/test_data_cleanup.py and test_surviving_process_recovery.py plus changed-contract registry/runtime-owner/authority/coordinator/recovery/continuation and F6 neighbors. Cover missing join, coherent matching-authority backup missing owner+joins, orphan roots, clean/dirty restart, write/readback failures, late launch/shutdown races, shared A/B control, idle/reader ownership, uncertain stop, stale PID, root/anchor replacement, persistence cuts and cancellation. Qualify separate-process unchanged-root/replacement behavior on actual local APFS; label Windows/external-writer guarantees unqualified. No full suite.
11. Check upgrade from exact final-F7 predecessor using populated retained roots and pending/idle/settled users; failed/read-only/newer/malformed migrations refuse without writes. Verify actual wheel resources if a migration is added. Run syntax/Ruff/Black/whitespace on changed Python with established shared-file baseline handling; only run startup tests if changed imports justify them.
12. Update Plugins/README.md and author/operation docs with data retention, review, recovery and cleanup status; write task-8-report.md with exact API/phase/marker/data-pin handoff, real commands/output, changed files and limitations. Controller commits, runs independent review, resolves findings, then checks all AC/notes/Done through CLI. Supply final owner contract to H2/M4/H6/I6/I7; no implementation overlap.
13. R46: follow f8-boot-identity-ruling.md. Bind the native Darwin boot-session UUID with exact native APFS device/inode/birth sec/ns identity; qualify unchanged-root application restart only within the same verified boot and all R43 clean/dirty/ownership checks. Changed, missing or malformed boot identity keeps retained roots fenced until explicit reviewed whole-root reconciliation/rebinding with generation advancement, never automatic takeover. Amend ADR-162/spec before source and document this operational limit. Use existing data_cleanup.py/runtime-owner/authority paths, no new dependency or production compiler; test actual separate-process native reads and same-boot service restart plus changed-boot/changed-incarnation/unavailable-native refusals, preserving sentinels and current authority. Simulations do not claim actual reboot qualification; I6 later shows the specific recovery reason.
14. Record the observed native fork-warning incident in backlog/docs/lessons-live-verification.md: same isolated stdlib versus sounddevice-only elevated control, CoreAudio/PortAudio sample, Python thread enumeration limitation, and fresh isolated warning-as-error fork verification preserving actual ownership assertions. Preserve all22 already-frozen implementation files and covering evidence; this documentation-only addition needs whitespace and an additional file hash, not repeated unchanged tests. Include the lesson in the task review package.
15. Address independent review F8-R1–R4 in one scoped fix round: completed-absence dirty recovery with ancestry/replacement refusal; seal ordinary mutation admission and settle queued work before final clean while preserving terminal callbacks; accumulate original epochs across all still-owned root grants; retain original review deadline through final completion. Add actual service/owner boundary controls in existing Tests/Plugins/test_data_cleanup.py, test_surviving_process_recovery.py and named runtime/authority/F6/F7 neighbors from task-8-fix1-findings.md. Preserve R43/R46, current source scope and one owner pipeline. Record focused RED/GREEN, frozen covering/static/hash evidence and append final Fix1 report before controller commit/scoped re-review; do not mark Done until every finding is resolved.
<!-- SECTION:PLAN:END -->
