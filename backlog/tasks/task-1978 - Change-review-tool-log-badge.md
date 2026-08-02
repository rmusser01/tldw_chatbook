---
id: TASK-1978
title: 'Change review: 'changed outside direct file tools' badge'
status: To Do
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - console
  - change-review
  - agents
dependencies:
  - TASK-1971
  - TASK-1973
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cross-reference the run's recorded steps (AgentRunsDB) with the turn's changed files: a file no recorded file tool touched gets a `⚠ changed outside direct file tools` badge in the tree — turning the B..E attribution limit into signal (script side effects and external writers become visible AS SUCH). Copy is exact: 'outside direct file tools', never 'not by the agent' — script writes are agent work too, and badge absence is not proof of tool provenance.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A write_file-modified file carries no badge; a file created by a script the agent ran carries the badge
- [ ] #2 Badge text matches the spec copy exactly and renders in monochrome
- [ ] #3 A run with no recorded steps (older data) renders without badges rather than badging everything
- [ ] #4 Badge derivation is tested against real recorded step shapes, not hand-built dicts
<!-- AC:END -->
