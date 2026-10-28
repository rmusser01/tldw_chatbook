---
id: TASK-37146
title: Distinguish unloaded tool schemas from unavailable tools
status: Done
assignee:
  - '@codex'
created_date: '2026-10-28 16:43'
updated_date: '2026-10-28 17:22'
labels: []
dependencies: []
documentation:
  - backlog/decisions/252-tool-availability-refusals-before-hook-preparation.md
  - Docs/superpowers/plans/2026-10-28-tool-disclosure-refusals.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console project inspection can stop after misleading Tool not permitted results even when a tool is allowed but its schema is absent from the current working set. Give the model a truthful recovery route while preserving existing permission, disclosure, and approval boundaries.

The reporting user was running Fedora Linux. These tool-discovery corrections are platform independent; Windows-specific findings from local verification are separate and do not explain that report.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Allowed tools without a loaded schema receive an actionable reload refusal, distinct from tools unavailable to the run, before any provider or skill executes.
- [x] #2 Unavailable or unknown tools retain a generic refusal without revealing catalog membership or suggesting that loading can grant access.
- [x] #3 Reloading an allowed tool recovers execution after initial deferral or working-set replacement on native and fenced tool-call paths.
- [x] #4 Ordinary catalog tools and skill tools use consistent refusal behavior, preserve blocked and not-started provenance, and do not invent an approval denial.
- [x] #5 Hook-enabled runs preserve the same availability refusals before hook preparation; unrelated preparation errors remain generic and fail closed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes; ADR path: backlog/decisions/252-tool-availability-refusals-before-hook-preparation.md; existing: backlog/decisions/104-token-budgeted-agent-tool-disclosure.md, backlog/decisions/078-structured-agent-tool-outcome-provenance.md, backlog/decisions/154-agent-denial-streak-boundary.md; Reason: routine correction of model-facing diagnostics under existing disclosure and permission contracts. 1. Add failing service-level refusal and reload regressions. 2. Share the narrow availability check across ordinary and skill dispatch without widening authority. 3. Run targeted tests, lint, format, and self-review; update architecture and task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented truthful, recoverable tool availability refusals without widening authority. A shared service check preserves the generic refusal for excluded/unknown names and directs allowed, registered but unloaded tools to load_tools with a retain-all reminder. Ordinary and skill dispatch share the check. Hook-enabled runs now carry a typed blocked/not-started refusal before preparation, preserving generic handling for arbitrary preparation exceptions and malformed results. A fixed ERROR prefix prevents refusal text from colliding with the reserved proceed verdict; refusal metadata never invents approval or denial facts.

Validation: test-first RED for ordinary/native/fenced/replaced-schema and skill cases, plus REDs for hook masking, unknown allow-listed names, and reserved-verdict text; all repaired. Final integrated targeted batch: 67 passed. Final guard/hooked-recovery batch: 10 passed. Targeted E9/F lint, seven-file formatting and whitespace checks passed. agent_service.py retains exactly its pre-existing formatter differences against HEAD.

Baseline qualification: the broader existing Windows hook/approval batch had the identical 19 failures and one teardown error on HEAD and on this patch (HEAD: 51 passed; patch: 56 passed including five new controls). An unchanged low-level command-hook test reports unsupported_platform; command hooks intentionally refuse Windows. These existing platform limitations were not bypassed. No full test sweep or live model was used. The reporting user's exact initial trigger remains unconfirmed without their run log.

Files: Agents/agent_service.py, Agents/agent_runtime.py, focused Agent tests, the living agent-runtime architecture reference, design/plan, and ADR-252. ADR-252 records the additive typed preparation contract; ADR-104, ADR-078 and ADR-154 authority/disclosure rules remain in force. The Windows identity issue encountered during local validation is separate from the Fedora report and is already fixed on dev by commit 6fe22b80ba.
<!-- SECTION:NOTES:END -->

Related repair: ADR-252 records the typed preparation refusal contract. The shared availability check also verifies catalog membership before promising reload recovery.
