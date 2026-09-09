---
id: TASK-32120
title: Expose native goal controls and qualify CLI workflow
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:19'
updated_date: '2026-09-09 15:04'
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
- [x] #1 Real Console controls start, pause, stop, resume and review the exact goal while preserving ordinary drafts and navigation lifetime.
- [x] #2 Canonical F9 Settings expose independent goal enablement and finite policy; setup states actual executor authority and verification scope.
- [x] #3 A real controller and agent path runs a trusted CLI verifier, corrects an editable fixture and passes the unchanged check across at least two increments.
- [x] #4 Controls and evidence remain usable at 80x24 and the ordinary terminal size; displayed actions and footer hints are implemented.
- [x] #5 Targeted integration, migration, privacy and diagnostics checks cover the delivered feature; documented live-provider evidence or an explicitly recorded unavailable prerequisite remains honest.
- [ ] #6 Every launch field shown at confirmation matches the submitted immutable request across awaited validation, and validation failure restores usable controls without stale selections.
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

Final whole-branch review fix wave (before new code): existing ADR-141 applies; no duplicate ADR. Reproduce the affected setup/selected-resource failures through isolated mounted/native requests, fix all findings in the shared final-review list while preserving owner boundaries, run focused amended regressions and scoped static checks, update user/qualification docs and commit. One independent scoped re-review follows. Root owns final AC/status/notes after approval.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Console goal controls under accepted ADR-141 (backlog/decisions/141-native-console-goal-runs.md): immutable setup from canonical owners, app-runtime-owned launch and restart Resume, exact checkpoint/run/evidence controls, bounded Older/Newer history and settled payload removal. Canonical F9 Console behavior saves independent finite [agents] goal policy atomically while preserving ordinary drafts and fleet settings.

The real controller/agent/SQLite/CLI fixture failed its trusted check (exit7), made an authorized fs_edit and passed the unchanged verifier (exit0) across two increments/five model calls. Production-style modal captures and mounted actions cover80x24/160x44. The configured local Qwen endpoint made two bounded attempts but did not call tools or correct the file; malformed/no-progress reports were refused. This is documented negative live evidence, not a successful model demonstration.

Independent Task5 specification and quality review approved implementation dd943d03143 and fix e98f6c68aa after allfour restart/catalog/summary/history findings were addressed. One nonblocking stale-validation form-state finding is carried into the final whole-branch fix wave and remains explicitly tracked in Docs/superpowers/reviews/2026-09-09-goal-runs-qualification.md.

Evidence:98 affected passes/1opt-in live skip,10selected controls passes,6setup passes; fix gate84passes and6strengthened followups. Counts overlap. Scoped Ruff/format and whitespace pass, with no added lint tuples in touched legacy owners. Architecture/privacy/migration gate97passed/3proven pre-goal diagnostic failures/1historical-commit skip; global diagnostics remain red. No full suite. Commands, actual traces, rendered controls, limitations and baseline proofs are retained in Docs/superpowers/qa/native-goals/task5/README.md and the qualification report. Main owners: UI/Console_Modules/goals.py, Console goal widgets/settings helper, Chat/console_goal_runs.py, existing runtime/skill/settings/command owners and targeted Tests/UI,Tests/Chat.

The requested successful live corrected artifact was not obtained from the configured model; the deterministic provider demonstrates the real CLI correction while the actual live failures remain preserved. Whole-branch review/integration reconciliation remains separate from this task-scoped approval. No new ADR beyond ADR-141; applicable implementation decisions and limits are linked in the qualification report.
<!-- SECTION:NOTES:END -->
