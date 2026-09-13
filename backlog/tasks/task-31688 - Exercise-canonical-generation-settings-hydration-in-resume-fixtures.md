---
id: TASK-31688
title: Exercise canonical generation settings hydration in resume fixtures
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:27'
updated_date: '2026-09-05 18:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve durable settings resume coverage after canonical generation hydration separated from the legacy prompt-only wrapper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both reported complete-snapshot and durable-replacement resume tests pass at the canonical hydration boundary
- [x] #2 Current provider endpoint configuration and row-owned prompt and prefill precedence remain asserted
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/095-conversation-owned-console-generation-settings.md
Reason: Test-only migration to the established canonical settings boundary.
1. Reproduce legacy-wrapper failures and read current hydration authority.
2. Retarget fixtures to canonical hydration and current endpoint ownership.
3. Run the complete hydration file plus generation-settings contract tests and static checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retargeted complete-safe-snapshot resume coverage to hydrate_console_generation_settings and the allowlisted metadata codec. Current configuration wins for endpoint and runtime character identity is not restored from generation metadata; canonical row prompt/prefill still win. Durable replacement now uses the real live settings submission plus serialized persistence commit before reopening, rather than the legacy store replacement that only writes compatibility metadata. Added canonical-owner persistence assertions while retaining legacy and sibling preservation checks. Existing ADR-095 applies; no new ADR. Two RED cases reproduced; complete hydration and metadata files plus related name-refresh tests:117 passed30.08s. Ruff lint, changed-block formatting and diff checks passed; self-reviewed.
<!-- SECTION:NOTES:END -->
