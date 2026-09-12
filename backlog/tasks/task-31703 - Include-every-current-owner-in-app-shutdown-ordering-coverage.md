---
id: TASK-31703
title: Include every current owner in app shutdown ordering coverage
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:49'
updated_date: '2026-09-05 18:57'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the shutdown-order fixture after new durable owners were added to the canonical app lifecycle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The fixture supplies every currently awaited lifecycle owner and asserts its exact authority order.
- [x] #2 The complete TTS app ownership file passes without production lifecycle changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the missing collections capture shutdown RED and inspect the entire canonical owner sequence. 2. Add explicit recording sentinels for collections capture, raw CLI, terminal sessions, and settings durability to the fixture and expected order. 3. Run the complete TTS ownership file, scoped checks, review, and commit. ADR required: no. ADR path: N/A. Reason: fixture-only coverage of existing owner order.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Expanded the explicit shutdown-order fixture with recording callbacks for collections capture, raw CLI, terminal sessions, and Console settings durability, matching the canonical awaited sequence. Preserved all existing owner ordering checks and added the four exact positions. All 42 TTS app-ownership cases passed within the clean 125-test fixture gate (/private/tmp/tldw-31702-31703-31705-first.xml). Scoped Ruff/format and diff checks passed. No production changes or new ADR required.

Parent reviewed the final scoped diff with no actionable findings.
<!-- SECTION:NOTES:END -->
