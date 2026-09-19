---
id: TASK-32777
title: >-
  Initialize Tool Profile authority before automatic workspace Persona
  provisioning
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 08:40'
updated_date: '2026-09-18 08:59'
labels:
  - ui
  - workspaces
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cold workspace creation currently leaves automatic Persona defaults unset because the Tool Profile guard is still deferred. Preserve lazy startup and existing admission while making the automatic choice reliable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cold automatic creation persists a real Persona and workspace permission-profile reference without opening Tool Profiles first.
- [x] #2 Pending startup backfill waits for complete Tool Profile authority and creates no orphan Persona while that authority is unavailable.
- [x] #3 Explicit None, saved Persona choices, duplicate-submit prevention, cancellation and nonfatal workspace creation semantics remain intact.
- [x] #4 Targeted regression and native private-profile evidence verify cold creation, persisted references and clean shutdown; unrelated startup remains lazy.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/079-workspace-assistant-defaults.md; backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
Reason: restore existing automatic provisioning using the existing deferred Tool Pack authority; no new admission policy, persistence contract, or startup budget.
1. Reproduce cold creation and pending backfill with the real deferred guard and private persistence.
2. Await existing app-owned Tool Pack composition before wiring/provisioning; initialize for eligible backfill only, keeping empty startup lazy.
3. Make the shared create modal await readiness without duplicate commits or writes after cancellation; preserve explicit None, saved choices and nonfatal automatic failure.
4. Run targeted lifecycle/guard tests and native cold creation with reopen/shutdown evidence, then independent review and save to draft PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cold automatic creation and eligible startup backfill now await the existing app-owned Tool Pack composition. Wiring refuses the inactive guard before creating Persona/profile records. The modal shields shared initialization from cancellation, rechecks ownership, and revalidates the form; explicit None and saved choices preserve their semantics.

Modified app.py, the shared create modal, ten real-storage UI regressions and five startup test wrappers. Original startup assertions are unchanged. ADR-079/097/139 apply; no new ADR or budget exception.

115 distinct targeted cases passed, two optional skips. Final native run002 (PID78000) passed four theme/size cells, first cold, with twelve inspected SVG/TXT pairs, exact reopened references, normal exit0, eleven healthy private DBs, no durable conversations/messages and unchanged default fingerprints. Independent review found no blocker. Ruff adds no app diagnostics; remaining preexisting diagnostics and corrected fixture failures are documented. QA: Docs/superpowers/qa/2026-09-18-workspace-cold-provisioning/README.md. Full suite and provider requests were not run. Historical failed rows after a completed backfill remain outside this bounded fix.

Persistent diagnostic inventory reviewed before refresh: app.py 404→406 calls, TASK-494 total7761→7763; the two new warnings contain only exception type names, no user content, paths, URLs or secrets. No sink topology changed. Inventory rebuilt with the existing --write command; statement review retained in the QA directory.
<!-- SECTION:NOTES:END -->
