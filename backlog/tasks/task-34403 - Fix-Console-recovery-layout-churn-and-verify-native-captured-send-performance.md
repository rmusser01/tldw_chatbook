---
id: TASK-34403
title: Fix Console recovery layout churn and verify native captured-send performance
status: In Progress
created_date: 2026-10-04 19:51
assignee:
- '@codex'
updated_date: 2026-10-04 22:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair measured root causes from TASK-34402 under the user-authorized combined performance fix pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Repeated unchanged recovery-row state produces no stylesheet work and visible/hidden geometry remains correct
- [ ] #2 Three captured messages complete with exact trace receipts on native Windows Linux and macOS with recorded performance counts
- [ ] #3 All identified fixes are reviewed and included in one PR against dev
- [ ] #4 Background scheduler safety checks resolve configuration and read the stop state off the UI thread while retaining fresh fail-safe and cancellation ownership behavior
- [ ] #5 Staggered subscription indexing performs blocking configuration and native admission only in its declared worker while retaining its fresh refusal behavior
- [ ] #6 MCP catalog setup keeps loop-bound asynchronous catalog and publication on the main loop while blocking store-backed reads run in finite workers with fresh permission resolution and safe cancellation
- [ ] #7 Each witness query uses one fresh validated control-record and registry observation, preserving source/startup/paired-generation refusals and rejecting changes during observation without authority reuse across calls
- [ ] #8 Coordinator locks never cover blocking native source validation; exact issued operation and native ownership are rechecked after out-of-lock validation and pause cancellation retarget and retirement refusals remain intact
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-10-04-console-performance-fixes.md. Reproduce RED, implement minimal fixes, verify targeted native and authority regressions. ADR check: UI batching N/A; native evidence and trace lifetime amend ADR-126/097 as needed.
Confirmed TASK-34403 AC8 custody follow-up: corrected native MCP fixture uses valid Windows TOML and seeds real permission bytes. Reproduce the admitted RMW pause failure against literal HEAD raw_participants.py and restore the fixed bytes in finally. Preserve the coordinator repair while using only the exact currently issued installed MCP raw source, its live thread/task/state/participant mapping and native leases to retain canonical generation observation custody. Acquire any distinct canonical observation lease before acceptance, never borrow a selected-generation lease for another path, and perform fresh witness checks on every use. Move remaining bound-source native checks in participant and scope methods outside the coordinator and recheck exact metadata, closure, pause and operation ownership before mutation/publication. Verify real accepted pause completion, no new acquisitions after pause, foreign same-path source/thread/task/sibling and native-demotion/retarget refusals, cancellation/native drain, plus existing four coordinator race controls. ADR required: yes; amend backlog/decisions/126-complete-local-backup-and-recovery.md for this finite retained canonical observation interface before production. No witness/permission cache, expanded lease authority, timer suppression or heavier app tests.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Finite fresh witness observation (ADR-126 AC7) is frozen: one validated control-record/registry pair per actual lease query; source/startup/paired rules remain fresh and completion checks detect inserted pending or changed/replaced native evidence. Valid RED2/11; final relevant native60pass/1POSIX-only skip56.28s; separate corrected CRLF edit control1pass8.70s. Paired witness119->89 nativeopens; unchanged surrounding installed MCP profile3003->2414 perkill, without cross-call witness/permission cache. Exact source hashes and qualified broader/baseline failures are recorded in the existing implementation plan. Formatter passes and Ruff adds no diagnostics. Native CI, independent review and whole-send budgets remain pending; task stays In Progress.
Independent witness completion review repairs are locally verified: Windows fresh ancestor DACL policy actualRED1failure->GREEN; POSIX bottom-up actual named-parent identity recheck added with real-platform ancestor rename control. Final relevant Windows61pass/2explicitPOSIXskips38.76s; paired native count remains119->89 (1record/1registry). Refrozen bootstrap SHA256624249b195edb3524b5bd44a78c2ec1f1bd3df00889f6f788d517991116a49ef. POSIX controls require real Linux/macOS CI; no platform spoofing or fabricated evidence. Existing plan/ADR126 receipt updated; combined task stays In Progress pending native matrix/whole-send proof.
Confirmed AC8 admitted MCP custody repair (ADR-126) is source/test frozen. Corrected fixture uses real installed configuration, valid Windows TOML and real True→False permission bytes; literal HEAD raw checker reproduces accepted RMW storage_locally_paused RED1/4.80s. Qualified actual paired-witness reader profiling gives original remaining participant/scope coordinator RED2/5.41s; exact real lease-validation barrier gives final source/canonical fence RED2/4.87s. Installed MCP raw operations now retain their exact canonical observation lease before acceptance, preserve exact lease execution_context path authority and re-read fresh witnesses; no new acquisition after pause and no witness/permission caching. Close/drain/resume use exact installed metadata without native I/O under the coordinator; all actual source proofs run outside it with issued source/thread/task/participant/native ownership rechecks. A refused same-path foreign call restores the already-issued outer operation before validating its continuation, never granting the foreign caller authority. Final formatted focused bundle21pass43.45s normalexit0 covers original four coordinator races, remaining entry proofs, maintenance closure/drain/resume, wrong canonical lease, final source/canonical mutations, real seeded RMW and same-path foreign refusal, installed/custom retarget/native demotion, foreign thread/task/sibling and actual native cancellation. Scoped production/new test Ruff GREEN, new test formatter GREEN, diffcheck GREEN; source-lifetimes diagnostics match HEAD. Frozen SHA256 raw451bde0e5a47aa7371612f8b349c45aed65c7e8067ef1263f9c9cc80e8d883e5, mcp_source809212856647086100231fb962d57983c05ae5d2b36c6ac5fdf690af95a9f770, recovery_activationf120eb3747210791ca513ab2c59fd75bf0f30cd7ed3c36d9ff3a41aca82d1b23, source_lifetimes78a0f3e735d858a0fa80968e5a048eb3ef9a96f3d3acb4828c479930150ce8ee, raw_coordinator119040923116b0aae7433651c781b7d680e794a8c6cefbf8f99c5f4f26f51ce2. Native matrix, mounted restore and whole-Console budgets still pending; task remains In Progress.
The distinct restored-generation custody branch is now qualified through an actual isolated restore and fresh MCP recovery review. Canonical and selected permission paths differ; their exact native leases are separately admitted before acceptance. A seeded real permission RMW continues through an actual local pause, finishes persisted policy bytes, and preserves all historical restored files. New test_restored_mcp_continuous_custody.py passes in 13.32s (normal exit 0), no OS or guarded callable substitution. Both paths remain freshly checked per issued operation, no authority cache. This adds no production changes after the raw source freeze and is included on all three native workflow hosts.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
