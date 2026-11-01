---
id: TASK-34665
title: Rebase PR3045 and resolve independent review findings
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-09 01:48'
updated_date: '2026-10-09 02:45'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/3045'
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qualify PR3045 against current dev and resolve independent Codex review findings before merging. User authorized independent review because Qodo is blocked by workspace credits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR is based on current dev and preserves later dev behavior.
- [x] #2 Each changed area is independently reviewed and every finding has a verified fix or technical disposition.
- [ ] #3 Targeted checks and derived-artifact guards demonstrate no introduced regressions; existing failures are attributed with evidence.
- [ ] #4 The reviewed head and verification summary are published and the PR is merged after required checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rebase onto latest dev, preserving additions and current dev routes.
2. Independently review DB/recovery/chat, provider/RAG, and character/UI/notes changes. Verify findings and repair shared owners with targeted regression tests.
3. Run changed-file tests and required derived-artifact guards. Compare existing failures against the base by message and cause.
4. Publish with explicit force-with-lease. Record review dispositions and merge only the verified head after required checks pass.
ADR required: no
ADR path: backlog/decisions/221-prompt-injection-cold-start-caches.md; backlog/decisions/222-provider-http-session-reuse.md; backlog/decisions/223-persistent-embedding-content-hash-cache.md; backlog/decisions/224-conversation-timestamp-normalization.md
Reason: review repairs implement the existing contracts. Create an ADR before changing any storage, ownership, or security boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased the 71-commit PR onto dev 4b1a256e880adf7d77a35e66a0d9222c77f01d27, preserving later dev semantic routes and both ADR/lesson additions. Qodo could not review because the workspace lacked credits; the user authorized an independent Codex review instead. Three reviewers covered the database/chat, provider/RAG and character/UI/Notes domains and cross-reviewed the repairs.

Repairs cover warm-cache admission, commit-safe prompt invalidation, native card revisions, registry-reader publication, incomplete Notes discovery, bounded tracking lookups, off-loop embedding-cache I/O, endpoint-separated vector keys, canonical timestamps, safe diagnostics and lazy provider/RAG imports. Tests include actual native SQLite, real threads and loopback HTTP; no paid provider was called. The duplicate world-info task ID is now 34666, preserving the earlier dev Console task 34415. Index-plan pins and the reviewed diagnostic inventory reproduce from source. Existing ADR-221 through ADR-224 apply; no new architecture decision.

Qualification is still in progress: the 56-file core selection completed with 1,696 passes and eight failures exactly reproduced on dev. Its introduced startup-census failure was repaired and the fresh startup/cache-ownership/connection-retirement selection passed all 27 tests. The UI selection completed with 572 passes; the remaining mounted UI cases are being compared to dev using the same admitted fixture. No full repository sweep was requested.

Latest-dev refresh: rebased the reviewed 72-commit series onto d880fc6d6b5731476a4a89352fa98825aca95fb8 after PR3047 merged. All 72 range-diff entries are unchanged, with no conflicts. The individually selected remaining UI cases produced identical outcomes and normalized causes on the fixed dev baseline and this PR: nine body failures, six passes, and six legacy config-fixture setup errors. The equivalent two-worker, three-file baseline comparison and relevant newly landed Notes integration tests continue before merge.
<!-- SECTION:NOTES:END -->
