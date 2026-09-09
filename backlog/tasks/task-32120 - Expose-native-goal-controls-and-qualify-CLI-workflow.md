---
id: TASK-32120
title: Expose native goal controls and qualify CLI workflow
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:19'
updated_date: '2026-09-09 08:19'
labels:
  - agents
  - console
dependencies:
  - TASK-32119
references:
  - backlog/decisions/141-native-console-goal-runs.md
documentation:
  - Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need to start and inspect autonomous goals in Console and see a real failed check become a verified corrected result.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Real Console controls start, pause, stop, resume and review the exact goal while preserving ordinary drafts and navigation lifetime.
- [ ] #2 Canonical F9 Settings expose independent goal enablement and finite policy; setup states actual executor authority and verification scope.
- [ ] #3 A real controller and agent path runs a trusted CLI verifier, corrects an editable fixture and passes the unchanged check across at least two increments.
- [ ] #4 Controls and evidence remain usable at 80x24 and the ordinary terminal size; displayed actions and footer hints are implemented.
- [ ] #5 Targeted integration, migration, privacy and diagnostics checks cover the delivered feature; documented live-provider evidence or an explicitly recorded unavailable prerequisite remains honest.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/141-native-console-goal-runs.md (Accepted)
Reason: Implements the accepted native goal UX/runtime ownership and private-evidence contracts, with canonical F9 Settings; no duplicate ADR.

1. Read the task, approved slice interfaces, relevant ADRs/lessons and existing Console module/settings owners. Add failing mounted tests for actual Start/Pause/Stop/Resume/review/removal/navigation and independent ordinary drafts.
2. Add a narrow runtime-owned metadata notification and minimal setup/status/control modules. Derive immutable launch bindings from existing trusted owners, retain stable launch IDs and exact checkpoint/artifact/native-run navigation, and keep the app runtime alive across view unmount.
3. Extend canonical F9 Console behavior's draft/Save/Revert path with independent finite goal policy persisted atomically in the actual [agents] section. Preserve manual/fleet semantics and the screen-size ratchet.
4. Extend the real controller/native/SQLite/CLI fixture through failed check, authorized file correction and unchanged passing check across at least two increments. Use model-visible evidence keys and verify artifacts, counters, diffs and external sentinels.
5. Inspect production-style rendered controls at 80x24 and ordinary terminal size; confirm focus, scrolling, wait reasons and exact review links. Use the verified local SVG renderer and one bounded visual correction pass.
6. Run a strictly finite isolated demonstration against the configured local model endpoint with the actual tool/report protocol; save and inspect commands/results/diff/final artifact, recording model or prerequisite limitations honestly without cloud fallback.
7. Run targeted new UI/integration and affected settings/runtime tests plus applicable migration/private-data/diagnostic/architecture checks. Classify baseline failures with evidence, update scoped inventories and user docs, run scoped lint/format/diff checks, self-review and commit. Root handles independent review and completion bookkeeping.
<!-- SECTION:PLAN:END -->
