---
id: TASK-13154
title: Supervisor agent fleet program
status: In Progress
assignee: []
created_date: '2026-08-09 13:57'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Named sub-agent definitions, background/parallel execution, steering, Console fleet panel. Spec: Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All six PR-phase subtasks Done
<!-- AC:END -->

## Deferred to later phases (final-review triage, 2026-08-09)

The final whole-branch review of PR-1 (named agent definitions) accepted
these as real but out of scope for the fix wave that landed the review's
findings. Recorded here because the SDD ledger that carried the full triage
(`.superpowers/sdd/2026-08-08-supervisor-fleet-pr1-agent-definitions/`) is
gitignored and does not survive merge.

- **PR-2a** — convert the spawn-closure disjoint-path `assert` to a `raise`
  (an `assert` is stripped under `python -O`, silently turning a real
  invariant violation into undefined behavior); add a load-once-per-turn
  call-count guard where `run_turn` changes.
- **PR-2b** — memoize/close the Settings ▸ Agents panel's `AgentRunsDB`
  handle. It is currently opened fresh on every category visit and relies
  on garbage collection to close the underlying connection rather than an
  explicit lifecycle.
- **Phase-4 polish** — give per-save feedback when `RUNTIME_TOOL_NAMES`
  entries are silently dropped from a typed tool list, so a user who lists
  `spawn_subagent` sees why it didn't stick.
- **Owner taste call** — where the Agents category belongs in Settings
  navigation (Troubleshooting vs. Expert) is a placement judgment call, not
  a defect; left for the owner to decide.
