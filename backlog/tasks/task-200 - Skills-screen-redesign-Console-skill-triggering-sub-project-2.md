---
id: TASK-200
title: Skills screen redesign + Console /skill triggering (sub-project 2)
status: To Do
assignee: []
created_date: '2026-07-12 13:16'
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
- [ ] #1 Own design spec approved before implementation,Skills CRUD + import wired in the redesigned screen honoring trust gating,Console can trigger a skill by slash command consuming SkillExecutionResult.rendered_prompt
<!-- AC:END -->
