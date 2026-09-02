---
id: TASK-27058
title: Inspect Tool Packs without side effects
status: Done
assignee:
  - '@codex'
created_date: '2026-09-01 00:00'
updated_date: '2026-09-02 04:33'
labels:
  - tool-packs
  - import
  - security
  - review
dependencies:
  - TASK-26070
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Inspect an untrusted Tool Pack into an immutable, expiring review without
extracting it or mutating permission/workspace authority, and classify only exact
portable contract matches or explicit one-to-one external-server mappings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Inspection reads one bounded regular archive through a no-follow descriptor, accepts only the exact pinned two-member stored ZIP envelope, and rejects hostile names, types, flags, headers, bounds, duplicates, compression, and nested or extra content.
- [x] #2 Manifest and payload admission uses strict canonical contracts and exact size/digest relationships; corrupt or unknown permission-store bytes remain byte-identical and legacy mutating recovery is never called.
- [x] #3 Destination ids are normalized without suffixing, reject reserved/invalid/exact or case-folded collisions, and reject any active or archived workspace reference including dangling references.
- [x] #4 Automatic matches require exact authority/server/raw tool name/portable contract fingerprint; external MCP mappings are explicit, capped, one-to-one, collision-free, and never use labels, projected names, fuzzy matching, configuration, or secrets.
- [x] #5 One strict store snapshot and one complete destination inventory produce an immutable 15-minute review that distinguishes exact matches, changed contracts, missing tools, pending Denies, and omitted Ask/Allow rules without any mutation callback; targeted tests and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing hostile-archive, bounds, strict-JSON, digest, no-follow, and live-store byte-preservation tests.
2. Implement bounded descriptor reads, exact ZIP header/member admission, and manifest-before-payload contract validation without extraction.
3. Add failing destination-id, profile/reference collision, exact identity/fingerprint, risk-tag change, disconnected-cache, mapping, and duplicate-result tests.
4. Implement immutable mapping/result types, strict one-snapshot destination validation, exact/manual classification, and 15-minute review expiry.
5. Run focused importer/safety tests, related contract/inventory tests, scoped Ruff, diff hygiene, self-review, and independent review.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already fixes side-effect-free strict inspection, exact/manual mapping, unbound destination admission, portable fingerprint semantics, privacy exclusions, and the later activation boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added descriptor-bound, no-follow, bounded admission for the exact canonical two-member stored ZIP envelope without extraction.
- Added strict manifest/payload validation, immutable expiring reviews, destination-id and workspace-reference collision checks, exact portable-contract matching, and capped one-to-one external MCP mapping.
- Kept inspection side-effect free by using one strict permission-store snapshot and one complete destination inventory; corrupt or unknown store bytes remain untouched.
- Independent review found an unresolved source-server Ask fallback was retained. The fix now omits missing Ask, retains restrictive Deny, and preserves required global/builtin plus exact or explicitly mapped Ask fallbacks; focused re-review approved the correction.
- Fresh verification passed 55 importer/safety tests and 151 contract/catalog/import tests, scoped Ruff, and diff hygiene. The only warning was the known environment-level Requests dependency warning.
- ADR check: no new ADR required; implementation follows `backlog/decisions/107-portable-tool-use-packs.md`. A Minor raw archive-path coercion hardening observation remains recorded for final review and does not affect the accepted path contract.
<!-- SECTION:NOTES:END -->
