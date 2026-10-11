---
id: TASK-32120
title: Expose native goal controls and qualify CLI workflow
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:19'
updated_date: '2026-10-11 22:37'
labels:
  - agents
  - console
dependencies:
  - TASK-32119
references:
  - backlog/decisions/227-native-console-goal-runs.md
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
- [x] #6 Every launch field shown at confirmation matches the submitted immutable request across awaited validation, and validation failure restores usable controls without stale selections.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/227-native-console-goal-runs.md
Reason: accepted native goals integrated with current storage, runtime custody and authority.
Follow TASK-34797 and Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md; preserve historical evidence, test targeted current owners and record fresh receipts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Console goal controls under accepted ADR-227 (backlog/decisions/227-native-console-goal-runs.md): immutable setup from canonical owners, app-runtime-owned launch and restart Resume, exact checkpoint/run/evidence controls, bounded Older/Newer history and settled payload removal. Canonical F9 Console behavior saves independent finite [agents] goal policy atomically while preserving ordinary drafts and fleet settings.

The real controller/agent/SQLite/CLI fixture failed its trusted check (exit7), made an authorized fs_edit and passed the unchanged verifier (exit0) across two increments/five model calls. Production-style modal captures and mounted actions cover80x24/160x44. The configured local Qwen endpoint made two bounded attempts but did not call tools or correct the file; malformed/no-progress reports were refused. This is documented negative live evidence, not a successful model demonstration.

Independent Task5 specification and quality review approved implementation dd943d03143 and fix e98f6c68aa after allfour restart/catalog/summary/history findings were addressed. One nonblocking stale-validation form-state finding is carried into the final whole-branch fix wave and remains explicitly tracked in Docs/superpowers/reviews/2026-09-09-goal-runs-qualification.md.

Evidence:98 affected passes/1opt-in live skip,10selected controls passes,6setup passes; fix gate84passes and6strengthened followups. Counts overlap. Scoped Ruff/format and whitespace pass, with no added lint tuples in touched legacy owners. Architecture/privacy/migration gate97passed/3proven pre-goal diagnostic failures/1historical-commit skip; global diagnostics remain red. No full suite. Commands, actual traces, rendered controls, limitations and baseline proofs are retained in Docs/superpowers/qa/native-goals/task5/README.md and the qualification report. Main owners: UI/Console_Modules/goals.py, Console goal widgets/settings helper, Chat/console_goal_runs.py, existing runtime/skill/settings/command owners and targeted Tests/UI,Tests/Chat.

The requested successful live corrected artifact was not obtained from the configured model; the deterministic provider demonstrates the real CLI correction while the actual live failures remain preserved. Whole-branch review/integration reconciliation remains separate from this task-scoped approval. No new ADR beyond ADR-227; applicable implementation decisions and limits are linked in the qualification report.

Final whole-branch review fix 525c61a944 and independent scoped re-review at 052fd4b95f address all five final findings with no new issues. All launch fields are frozen before awaited validation, final Start persists the displayed request and rejected/stale validation restores only current enabled controls. Mounted all-field and cleanup regressions verify the service result.

Verification: 279 affected passes, 1 intentional live skip and 1 proven unchanged missing-run_id test failure. The 20 strengthened setup/source passes and 2 geometry/2 cleanup followups overlap. Scoped static checks pass with zero added legacy diagnostics. Deterministic real CLI execution still exits 7→0 across 2 increments and 5 calls with an unchanged verifier and the inspected corrected file/diff. Three separate baseline diagnostic failures and unsuccessful historical local-model trials remain disclosed; no full suite or further model calls.

ADR required: yes. Existing accepted backlog/decisions/227-native-console-goal-runs.md was clarified before this fix; no new stored fields, schema or execution/permission owner. Durable final review, exact evidence and limits: Docs/superpowers/reviews/2026-09-09-goal-runs-final-review.md and Docs/superpowers/reviews/2026-09-09-goal-runs-qualification.md, with QA under Docs/superpowers/qa/native-goals/task5/.

2026-10-11: reopened for current-dev integration under TASK-34797 and ADR-227. Prior notes and counts describe historical source only; they do not certify the current integration.

2026-10-11: reopened for current-dev integration under TASK-34797 and ADR-227. Prior notes are historical source qualification only.
<!-- SECTION:NOTES:END -->
