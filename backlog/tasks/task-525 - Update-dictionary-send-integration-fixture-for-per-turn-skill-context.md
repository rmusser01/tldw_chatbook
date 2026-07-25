---
id: TASK-525
title: Update dictionary send integration fixture for per-turn skill context
status: Done
assignee: []
created_date: '2026-07-24 19:04'
updated_date: '2026-07-24 19:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the native dictionary-send agent integration test aligned with the agent bridge contract so the test reaches and verifies conversation-dictionary injection instead of failing inside an outdated fake.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The fake agent bridge accepts the current per-turn skill binding and bundle arguments
- [x] #2 The agent-branch test captures and verifies injected agent messages
- [x] #3 The full dictionary-send integration module passes
- [x] #4 The merge-base result and no-ADR decision are documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the feature-branch failure and inspect the current bridge call contract.
2. Update only the fake bridge signature with the explicit current optional arguments.
3. Run the full dictionary-send integration module, Ruff, diff checks, and independent review.
4. Document the merge-base result, no-ADR decision, and implementation notes before completion.

ADR required: no
ADR path: N/A
Reason: This is a test-fixture contract update and changes no production API, storage, or application architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the agent-branch fake ConsoleAgentBridge.run_reply signature with the current optional turn_skill_bindings and turn_bundle_block arguments. The explicit parameters preserve contract checking while allowing the fake to capture the dictionary-substituted agent_messages payload.

Before the fix, the fake rejected the new keywords inside asyncio.to_thread; the controller converted that TypeError into an accepted failure result, leaving the capture empty and causing the test to fail with KeyError. The merge-base run is masked earlier by its known production-path sqlite readonly failure, now repaired by TASK-522. Verification: both dictionary-send provider and agent tests pass; the combined TASK-523/524/525 batch passes 79 tests; Ruff, format, and diff checks pass. Independent review approved the narrowly scoped fixture update.

ADR required: no. This aligns a test double with the existing agent bridge contract and changes no production API, storage, or architecture.
<!-- SECTION:NOTES:END -->
