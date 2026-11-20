---
id: TASK-42071
title: "Burn-down lane additions red on Linux CI: runner Git LFS filters, fleet race, RAG extras"
status: Done
assignee: []
created_date: 2026-11-19 23:30
labels:
- follow-up
- tests
- ci
priority: high
references:
- Tests/Chat/test_console_reference_custody.py
- Tests/Chat/test_change_turn_tracking.py
- Tests/UI/test_console_send_without_mcp_store.py
- tldw_chatbook/Workspaces/native_git.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TASK-41630..41655 burn-down's lanes commit (527296c14b) put `Tests/Chat/test_console_reference_custody.py` and `Tests/Chat/test_change_turn_tracking.py` into a CI lane for the first time. It also added a new UI test. That lane's runs were cancelled on the stack, so the additions first ran on Linux CI after landing. There, 7 nodes failed, and dev a6df4d8d48's required check turned red. All 7 passed on macOS. Found by the #3637 run (head 00d2c77e11): admission-sensitive 4, job 135948664270, and UI Fast Lane 19, job 135948665513.

There were three causes, all specific to the test environment:
- **Git LFS filters (4 reference-custody nodes).** GitHub's runners install Git LFS filters in the system Git config. Native change review refuses any configured filter ("configured filters"), and the product scrubs `GIT_*` from git's environment, so no environment variable can switch it off.
- **Fleet race (2 turn-tracking nodes).** The two negative controls assumed their quick child sub-agent would finish before the parent's final model call. On the runner, the child's model call started 12 ms after the turn settled, so it counted as a survivor (`subagent_post_turn`).
- **RAG extras (1 UI node).** The MCP-row height test pinned a card height of 17, measured with "RAG: Ready". The UI lane has no embeddings extras, so the row reads "RAG: Unavailable - Install embeddings.", wraps, and the card measures 18.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The four reference-custody nodes pass when the host's system Git config defines Git LFS filters, and the production filter refusal still runs
- [x] #2 The two fleet negative controls pass when the child sub-agent starts late (red first under a delayed child)
- [x] #3 The MCP-row fit test passes with and without the embeddings extras installed (red first with RAG unavailable)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (test isolation and a behaviour-identical helper extraction).

1. Reproduce each failure locally: a `git` shim that sets `GIT_CONFIG_SYSTEM` to an LFS filter file, a 0.5 s delayed child, and RAG forced unavailable.
2. Isolate host system Git config at a single test seam, following the precedent in `Tests/UI/test_console_session_admitted_roots.py`.
3. Make the fleet controls wait for the child thread before the parent's final call.
4. Pin RAG availability in the height test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **Git LFS filters:**
  - The scrubbed environment was built in three identical places: `ShadowRepo._env`, `ReferenceAuthority._capture` and `run_git_reference`'s default. It now comes from a single `native_git.scrubbed_git_environment()`, with no change in behaviour.
  - An autouse fixture in the reference-custody file adds `GIT_CONFIG_NOSYSTEM=1` to that one seam. The product's filter qualification still runs against user and repository config.
  - `console_references.py` no longer needs `import os`.
  - Under the LFS shim, the 4 nodes went from red to green, and the 143 nodes of both files pass with and without the shim.
- **Fleet race:**
  - `_FleetSurvivorGateway(parent_joins_children_on_call=N)` joins the fleet threads, through the executor, before parent call N. The two negative controls pass N=3.
  - With a 0.5 s delay before the child's first call, both reproduced CI's `['subagent_post_turn'] == ['turn']`, and both pass with the fix.
- **RAG extras:**
  - The test pins `ChatScreen._console_rag_extras_available` to True.
  - Forcing it False reproduced CI's `assert 18 == 17`. The file passes 18/18.
- **Review (local, replaces the bot review):** no Critical or Important findings. All three Minor items are fixed:
  - The stale 39-column comment on the RAG row in `Chat/console_live_work.py` now states the measured 36 columns and the 18-row card.
  - The gateway records `unjoined_children` after its join, and both controls assert it is empty. Delaying the child 6 s now fails with `['fleet-…']` rather than a misleading survivor row.
  - The gateway comment now cites TASK-42071.
- **Follow-up filed:** the reviewer saw a load flake in another newly laned node of the UI file. It is filed as TASK-42072.
- **Lint:** `ruff check tldw_chatbook --select F` passes. `Tests/lint/ruff_gate.py` reports only dev's existing B023 at `change_review_consent.py:550`, which #3637 fixes.
<!-- SECTION:NOTES:END -->
