---
id: TASK-32520
title: 'Docs: agent-runs-and-tools.md carries three merge-conflict markers on dev'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-13 02:55'
updated_date: '2026-09-29 19:28'
labels:
  - docs
  - console
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Docs/User_Guide/console/agent-runs-and-tools.md` on dev 7159fc0b99 contains
three unresolved merge-conflict markers — `>>>>>>> 31da72f8a5 (feat: agent
chat fork & spawn tools (fork_chat / new_chat) — rebased onto dev)` at line
470, `<<<<<<< HEAD` at line 1249 and `=======` at line 1455 — left by commit
6a6242ce3a (PR #2643, agent chat fork & spawn). The closing `>>>>>>>` at
470 sits above the `<<<<<<< HEAD` at 1249, and the `=======` at 1455 is
followed by "### Project instructions before tools run" with no closing
marker after it — so the markers do not pair, and which text belongs to
which side is not decidable from the file; a reader sees the markers as
text. Found by the wave-3 docs sweep's conflict-marker sweep (task-32271);
outside the Library ▸ Notes scope, so filed rather than fixed there.

Evidence: `git show origin/dev:Docs/User_Guide/console/agent-runs-and-tools.md
| grep -nE '^(<<<<<<<|>>>>>>>|=======$)'` → 470, 1249, 1455 on 2026-09-13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The page contains no conflict markers and reads as one document, with both sides of the fork/spawn conflict reconciled rather than one dropped
- [x] #2 A guide-consistency test (or the existing conflict-marker sweep in CI) fails on any User_Guide page that carries a conflict marker, so this cannot land again
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify the repaired guide retains fork and spawn documentation. 2. Add a User Guide conflict-marker guard and prove it detects a temporary marker. 3. Run the targeted docs check and static checks, then document evidence. ADR required: no; ADR path: N/A; reason: documentation regression guard only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The current guide already retains both fork_chat and new_chat sections with the historical markers removed. Added Tests/Docs/test_user_guide_conflict_markers.py covering every User_Guide Markdown page. A temporary actual conflict marker made the check fail at the expected guide line; the original bytes were restored in finally and the guard passed. New-file Ruff lint and formatting pass. ADR required: no, documentation guard only. Evidence: /private/tmp/agent-burndown-docs-green.log. Final guard and independent review pending after concurrent guide updates.

Final disposition 2026-09-29: independent read-only implementation review approved the scoped repair with no actionable findings. The targeted acceptance checks and changed-line static checks recorded above pass; inherited whole-file lint/format debt remains outside this correctness task. All acceptance criteria are checked and this task is Done. No full-suite or live-provider qualification is claimed. This disposition supersedes earlier pending-review notes.
<!-- SECTION:NOTES:END -->
