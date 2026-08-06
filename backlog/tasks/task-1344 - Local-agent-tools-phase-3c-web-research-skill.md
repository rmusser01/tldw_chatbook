---
id: TASK-1344
title: 'Local agent tools phase 3c: web-research skill'
status: Done
assignee: []
created_date: '2026-08-05 22:07'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md §2.6. Plan: Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3c.md. Includes skill-runner local-tool narrowing (disclosed spec deviation).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skill-run subagents can be narrowed against local tools (never granted beyond the parent's allow-list)
- [x] #2 A skill declaring web_search/web_fetch in allowed-tools gets exactly those (plus requested builtins). AMENDED (per backlog rules, before checkoff): undeclared skills now pass the full builtins+local set through to the child run (was "behave as before") — intentional, approval-gated, matches native spawn_subagent inheritance; documented in code, tests, and the install docs
- [x] #3 web-research skill definition parses and passes trust scanning
- [x] #4 Install documentation exists
- [x] #5 All new tests pass
<!-- AC:END -->


## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3c.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented on branch `feat/local-agent-tools-p2` (stacked on PRs #1352/#1358) via subagent-driven development with per-task spec + quality review.

- `Chat/console_agent_bridge.py`: `_BridgeSkillRunner` gains `local_names`; `run()` narrows via `intersect_skill_tools(declared, builtin_names + local_names)` — intersect-only, never grants (skill/MCP/runtime names cannot survive), backward-compatible default `()`. Construction site passes the per-run `local_names` (single source of truth, same tuple used for collision filtering).
- `Agents/tool_catalog.py`: `intersect_skill_tools` docstring generalized (the narrowing set is no longer builtins-only). `Agents/agent_service.py`: stale "builtins-only" comment corrected.
- `Docs/Examples/skills/web-research/SKILL.md` (new): the web-research orchestration skill — decompose into 2–5 sub-questions, `web_search` per angle, `web_fetch` primary sources, synthesize with inline citations, mandatory conflicts/caveats, never-fabricate rule, find_tools/load_tools discovery note, stop conditions, and an untrusted-content guard (fetched pages are data, never instructions). `Docs/Examples/skills/README.md` (new): requirements, verified install path (`get_user_data_dir()/skills/skills/`), and a skill-author note documenting the undeclared-skill capability expansion.
- `Tests/Skills/test_web_research_skill.py` (new): parses via the real public `import_skill`/`execute_skill` path, allowed-tools asserted against the registered default specs, trust scan clean. `Tests/Chat/test_console_agent_bridge_local.py`: 4 narrowing tests including a wire-level approval-gating e2e (child's web_fetch hits the shared review hook exactly once).

Spec deviation (disclosed in the plan): §2.6 said "no new runtime code", but skill children were narrowed against builtins only, so the skill's child would have had no web tools at all — Task 1's ~15-line widening is the minimal honest fix, matching native spawn inheritance.

AC #2 amended before checkoff (backlog rules): implementation makes undeclared skills pass builtins+local through (behavior change, documented in three places); the original "behave as before" text was literally false and could not be silently checked.

Tests: 422 passed (Agents+Skills), 1095 passed (Chat, deselecting the known pre-existing anthropic failure). Final whole-phase review: Ready to merge after the docs commits landed.
