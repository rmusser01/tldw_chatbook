---
id: TASK-200
title: Skills screen redesign + Console /skill triggering (sub-project 2)
status: Done
assignee: []
created_date: '2026-07-12 13:16'
updated_date: '2026-07-17 15:22'
labels:
  - ux
  - skills
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Second spec of the user-confirmed decomposition (Prompts -> Skills -> MCP). Redesign the Skills screen to Console-parity CRUD (view/review/modify/create/delete SKILL.md-based skills, wire the disabled Import, respect the trust boundary from ADR-009) and trigger skills from the Console composer via /skill <name> and bare /skill-name using the command-grammar fallback-resolver hook shipped by the Prompts spec. Requires its own brainstorm/spec before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Own design spec approved before implementation,Skills CRUD + import wired in the redesigned screen honoring trust gating,Console can trigger a skill by slash command consuming SkillExecutionResult.rendered_prompt
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped across the Skills program PRs (all merged to dev): spec Docs/superpowers/specs/2026-07-14-skills-library-console-design.md approved before implementation; Library ▸ Skills workbench (rail row/count, trusted+blocked populations w/ trust glyphs, SKILL.md detail editor + trust panel, import, tab retirement + route alias) and Console/agent invocation (/skills command + bare /skill-name fallback resolver consuming SkillExecutionResult.rendered_prompt, run allowlist composition, SkillToolProvider w/ per-run spawned executor, catalog dedupe/owner-map caching fixes) — landed via PR #636 (with #620/#623/#629 prerequisites). Trust gating honored end-to-end (trust store gates list/execute; agent permission checkpoint). Live gates: Docs/superpowers/qa/skills-library-2026-07/ and skills-console-2026-07/. This status edit fixes drift only — the work itself merged 2026-07-14/15.
<!-- SECTION:NOTES:END -->
