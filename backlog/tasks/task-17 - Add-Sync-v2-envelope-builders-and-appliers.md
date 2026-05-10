---
id: TASK-17
title: Add Sync v2 envelope builders and appliers
status: Done
assignee: []
created_date: '2026-05-10 15:14'
updated_date: '2026-05-10 15:18'
labels:
  - sync
  - client
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Chatbook-side Sync v2 envelope builders, appliers, and first-pass local domain adapters so local notes, chat messages, workspace source refs, source cache entries, and media compatibility records can be represented as Sync v2 envelopes and applied without overwriting conflicting local encrypted content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Note builders place private note title/body content into encrypted payloads while leaving only safe routing metadata clear
- [x] #2 Chat message builders use stable message IDs so repeated pulls can append idempotently and conflicting hashes are recorded
- [x] #3 Workspace source refs map add and remove operations to link and unlink envelopes
- [x] #4 Source cache envelopes use source ID plus content hash as their stable identity
- [x] #5 Appliers record local conflicts instead of overwriting divergent encrypted content edits
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing builder and applier tests with small fake local stores. 2. Implement SyncEnvelopeBuilder using the existing Sync v2 schemas and crypto helpers. 3. Implement SyncEnvelopeApplier plus small domain adapter modules for notes chat workspaces source_cache and media compatibility. 4. Run focused builder/applier/scope tests and the broader Sync_Interop sync API suite. 5. Run security and diff checks, update Backlog, and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented first-pass Chatbook Sync v2 envelope builders and appliers. Added SyncEnvelopeBuilder for encrypted note content, chat messages with stable conversation/message IDs, workspace source link/unlink envelopes, source cache entries keyed by source ID plus content hash, and media compatibility envelopes. Added SyncEnvelopeApplier and small domain adapters for notes, chat, workspaces, source_cache, and media; adapters route through local store hooks and record conflicts instead of overwriting divergent encrypted note content or chat message hashes. Verification: red tests first failed on missing builder/applier modules; target builder/applier/scope tests passed with 15 tests; broader Sync_Interop plus sync API tests passed with 57 tests; Bandit on new production files had 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Chatbook Sync v2 envelope builders and local appliers for the first local-first domains. Private note/chat/source-cache content is encrypted into Sync v2 payloads, workspace source membership maps to link/unlink envelopes, source cache identity uses source ID plus content hash, and pulled envelopes are applied through small local adapters that preserve local conflicts rather than overwriting divergent encrypted content.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit or equivalent security scan run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
