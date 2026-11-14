---
id: TASK-37631
title: Support virtual CLI listing flags and disclose command syntax
status: Done
created_date: 2026-10-29 22:34
assignee:
- '@codex'
updated_date: 2026-10-29 23:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let models use common read-only listing flags and learn the supported virtual command grammar without repeated invalid calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Virtual ls accepts common combined all and long flags with meaningful output through the pinned worker.
- [x] #2 The model sees supported command usage and invalid arguments return actionable syntax.
- [x] #3 Folder confinement, sensitive exclusions, independent permissions and bounded worker execution remain enforced.
- [x] #4 Focused tests, lint and relevant documentation pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/254-cli-search-outcomes-and-listing-compatibility.md. Reason: refine search outcomes and optional listing worker arguments while retaining ADR-078/094/101 authority. Follow Docs/superpowers/plans/2026-10-29-cli-tool-recovery.md with focused red/green tests, implementation, documentation, targeted checks and fresh review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Virtual ls accepts all/long aliases and combined -la/-al; long rows show type/permissions and byte size for admitted entries. Shared usage is visible in the schema and parser/schema errors. Optional strict long_format is forwarded through the closed local/remote worker protocol, with absent/false output compatibility. Generated remote worker rebuilt and checked. Evidence: virtual-bundle-green 275 passed, including real worker and shipped-bundle exclusion controls. Scoped new-code lint and formatting pass; inherited F821/E721 findings are unchanged. Optional HOME/symlink/profile controls showed identical eight setup failures with pristine HEAD listing functions; no assertions or authority checks weakened. ADR-254, user guide and architecture docs updated.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
