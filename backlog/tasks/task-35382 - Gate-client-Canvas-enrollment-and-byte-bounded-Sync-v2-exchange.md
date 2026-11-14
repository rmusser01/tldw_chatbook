---
id: TASK-35382
title: Gate client Canvas enrollment and byte-bounded Sync v2 exchange
status: To Do
created_date: 2026-10-18 05:57
dependencies:
- TASK-35379
- TASK-35381
labels:
- canvas
- sync
- client
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow deliberate Canvas source portability only after the complete approved server contract and local storage capability are available. Reuse existing Sync v2 policy and retries without broadening runtime privileges.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Explicit durable-conversation enrollment discloses authenticated-home-server readability under server_trusted_v1 and requires complete message-tree, lifecycle, storage, domain/version and effective-limit negotiation; missing/partial/malformed capabilities leave Canvas local while ordinary chat can continue.
- [ ] #2 Outgoing canonical envelopes preserve exact IDs/hashes and scoped base metadata, send at most one source-bearing envelope per request initially, and bound actual payload/request/page bytes without reducing artifact quotas.
- [ ] #3 Only matching applied or proven idempotent receipts drain outbox entries; uncertain outcomes retry exact identities, partial successes remain independent, and auth/identity/enrollment changes pause or fence exchange safely.
- [ ] #4 Older-client coexistence, new-domain baseline replay, unsupported profile/schema, retry-after, byte-limit and source-privacy tests pass; synchronization never installs runtime code, opens a preview or imports browser/file/network grants.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
