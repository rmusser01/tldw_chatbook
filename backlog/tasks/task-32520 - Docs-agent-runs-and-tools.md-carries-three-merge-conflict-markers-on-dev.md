---
id: TASK-32520
title: "Docs: agent-runs-and-tools.md carries three merge-conflict markers on dev"
status: To Do
assignee: []
created_date: '2026-09-13 02:55'
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
- [ ] #1 The page contains no conflict markers and reads as one document, with both sides of the fork/spawn conflict reconciled rather than one dropped
- [ ] #2 A guide-consistency test (or the existing conflict-marker sweep in CI) fails on any User_Guide page that carries a conflict marker, so this cannot land again
<!-- AC:END -->
