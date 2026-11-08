---
id: TASK-39403
title: Preserve ToolResult subclass positional compatibility
status: In Progress
created_date: 2026-11-08 22:07
assignee:
- '@codex'
labels:
- release
- agents
priority: high
references:
- TASK-18929
- TASK-39402
documentation:
- backlog/decisions/154-agent-denial-streak-boundary.md
- backlog/decisions/163-expanded-console-hook-runtime.md
updated_date: 2026-11-08 22:07
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Optional dispatch and termination metadata shifted the existing fifth positional argument in ToolResult subclasses. Restore the accepted four-field base positional contract without changing metadata values, authority or serialization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing ChildResult fifth positional construction binds its subclass field, while ok/content/error/outcome remain positional.
- [x] #2 dispatch_state and termination_reason remain available by keyword with unchanged types, defaultNone and serialized field order.
- [x] #3 Existing denial-authority, dispatch/cancellation, subclass and replay focused tests pass without security or resource-owner changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR.
ADR path: backlog/decisions/154-agent-denial-streak-boundary.md and backlog/decisions/163-expanded-console-hook-runtime.md.
Reason: compatible optional metadata and dispatch ownership repair; no storage, permission, provider-wire or runtime contract change.
1. Use authentic baseline010 provenance4passed/1failed and current574-constructor review; metadata producers already use keywords.
2. Change only ToolResult.dispatch_state and termination_reason to field(default=None, kw_only=True), reusing the field import.
3. Run the unchanged provenance file and13 reviewed focused selectors in a private offline profile; require exit/report agreement and natural process retirement.
4. Check exact AST/serialization-preserving diff, scoped lint/format and self review; record evidence, commit and push to PR3475.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Restored the four-base-positional/subclass contract by making only dispatch_state and termination_reason keyword-only, preserving defaultNone, annotations, field order and authority. The574-constructor audit confirmed metadata callers already use keywords. Original provenance evidence was4passed/1failed; the unchanged13-selector focused scope now passes25 parametrized cases, all setup/call/teardown phases passed, exit0, exact JSON/JUnit identity agreement and natural unforced owned-process retirement. Source stayed unchanged during the run. Exact two-field AST diff and scoped formatting pass. Ruff adds no diagnostics versus five pre-existing unrelated findings. No callers, tests, serialization, permission or resource-owner changes. Existing ADR154/163 apply; no new ADR. Report SHA256:1a13c2e40692205343d89583d00d5596128d8271c15f0f17879c1b8e2b94b392. Final integrated qualification remains open under TASK39402 and Docs/QA/release-0.2.4/BLOCKERS.md. Backlog MCP final-status operation temporarily failed to acquire its project lock; implementation remains reviewable while that operation is retried.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
