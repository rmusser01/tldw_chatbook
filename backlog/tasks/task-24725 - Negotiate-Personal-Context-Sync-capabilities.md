---
id: TASK-24725
title: Negotiate Personal Context Sync capabilities
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 18:43'
updated_date: '2026-08-30 19:36'
labels:
  - personal-context
  - sync
  - security
dependencies:
  - TASK-24723
references:
  - >-
    backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
documentation:
  - Docs/superpowers/plans/2026-08-28-personal-context-04-sync-multidevice.md
  - IMPLEMENTATION_PLAN_personal_context_sync_capabilities.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Require Chatbook to parse and validate the complete server Personal Context Sync v2 capability contract before linking or write synchronization is enabled, while leaving existing Sync domains unaffected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chatbook strictly parses the typed Personal Context capability object and tolerates only forward-compatible unknown fields.
- [x] #2 Readiness requires all five Personal Context domains, compatible schema bounds, server_trusted_v1, HMAC-SHA-256 integrity, wrapped bootstrap, cleanup acknowledgments, purge generations, and required quotas.
- [x] #3 Missing, malformed, downgraded, or incompatible capability data disables Personal Context reads/writes with stable blockers without affecting existing Sync domains.
- [x] #4 The API client and server sync service expose negotiated schema/readiness through one bounded service seam.
- [x] #5 Targeted Sync_Interop and API-client tests plus static, security, diff, and independent review gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing Chatbook capability parsing and readiness tests for complete, missing, malformed, downgraded, future-field, and incompatible server contracts.
2. Extend Sync v2 client schemas and API parsing with one typed Personal Context capability object.
3. Add a bounded readiness result in ServerSyncService and sync_readiness without changing existing-domain behavior.
4. Run targeted Sync_Interop/API-client regressions, static/security gates, independent review, update documentation/task evidence, and commit.

ADR required: no (existing)
ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
Reason: ADR-102 already governs Personal Context Sync domains, capability gating, integrity, cleanup acknowledgments, and purge generations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added strict, forward-compatible parsing for the typed Personal Context capability contract and bounded operation/adapter maps. Malformed Personal Context entries fail closed with a stable blocker, while malformed unknown future entries are ignored so unrelated Sync domains remain usable.
- Added one readiness seam that requires the exact five-domain set, upsert/tombstone operations, schema v1, `server_trusted_v1`, HMAC integrity, wrapped bootstrap, cleanup and purge contracts, quotas, and supported/writable adapter v1 maps.
- Added two-phase negotiation: unscoped capabilities must support reads before device/dataset work, then dataset-scoped capabilities must advertise writability before the dry-run push/pull. Partial dotted Personal Context selections are rejected before remote mutation, and negotiated readiness is persisted/exposed by the existing service.
- Verification: 81 targeted Sync_Interop/schema/API-client tests passed; Ruff, Python compilation, Bandit, and `git diff --check` passed. Test output retained the environment's existing Requests dependency warning and pytest temporary-directory cleanup warnings.
- Independent review identified and verified fixes for adapter/operation validation, pre-mutation domain atomicity, malformed-entry isolation, and the legacy dictionary `supported_operations` alias; final re-review returned CLEAN. The full repository suite was not run under the repository's targeted-test policy.
- ADR required: no. Existing ADR `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md` governs the implemented contract.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chatbook now enables Personal Context sync only after the complete shared server contract is negotiated, while preserving existing Sync behavior and legacy capability aliases. All scoped verification and review gates passed with no known implementation blockers.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Targeted tests and verification recorded
- [x] #3 Documentation updated
- [x] #4 Static and security checks pass
- [x] #5 Independent review completed
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
