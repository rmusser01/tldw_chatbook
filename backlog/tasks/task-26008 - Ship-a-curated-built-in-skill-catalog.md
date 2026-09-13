---
id: TASK-26008
title: Ship a curated built-in skill catalog
status: To Do
assignee: []
created_date: '2026-08-31 15:44'
labels:
  - skills
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Chatbook has a hardened skill runtime and ships no skills. Verified on origin/dev: find over the repo for SKILL.md returns 7 files, all under Tests/fixtures/ and Docs/Examples/ - nothing installed by default. The runtime is genuinely good (sandboxed script execution at Skills_Interop/skill_script_runner.py:1-31, trust-gated install requiring human review at Agents/tool_catalog.py:282-300) and the frontmatter is Claude-Agent-Skills-spec compatible (Skills_Interop/local_skills_service.py:505-580), so the format work is done. Hermes ships 398 skills across two catalogs. This task is content plus a discovery path, deliberately NOT a hub or registry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A small curated set of skills ships with the application and is discoverable without a manual install step
- [ ] #2 Built-in skills are visibly distinguished from user-installed ones and cannot be silently overwritten by an install
- [ ] #3 Built-in skills carry the same trust state semantics as any other skill; shipping one does not bypass the review gate for scripts
- [ ] #4 Each shipped skill is exercised by a test that at minimum parses its frontmatter and validates declared tools resolve
- [ ] #5 The set is small and defensible - each skill's inclusion reason is recorded, rather than bulk-importing a catalog
- [ ] #6 Uninstalling or disabling a built-in skill is possible and persists
<!-- AC:END -->
