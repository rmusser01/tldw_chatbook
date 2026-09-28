---
id: TASK-33165
title: Reserve Canvas context in the Personal Context Next Send preview
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-28 05:52'
labels:
  - memory
  - canvas
  - context-budget
dependencies:
  - TASK-25907.23
references:
  - tldw_chatbook/Chat/console_chat_controller.py
  - tldw_chatbook/Chat/console_agent_bridge.py
documentation:
  - >-
    backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md
  - backlog/decisions/186-dependency-aware-personal-context-forgetting.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When Canvas tools are enabled and the model input budget is tight, the Personal Context Inspector can report records as selected even though the dispatched request omits them. Make the preview reflect the complete request budget while preserving the disposable nature of inspection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A native production preview regression reproduces the Canvas-enabled budget mismatch before the fix.
- [ ] #2 With identical profile, model, message and tool inputs, the Inspector and dispatched request agree on selected records and their available input budget, including Canvas schemas and runtime guidance.
- [ ] #3 Preview does not register an executable Canvas run, create staging records, grant mutation authority, or change the selected Canvas.
- [ ] #4 Disabled, unavailable and stale Canvas states preserve their existing fail-closed behavior, with native Python >=3.12 targeted verification and a recorded ADR check.
<!-- AC:END -->

## Review Evidence

TASK-25907.23 current-dev refresh review traced the production controller preview into `ConsoleAgentBridge.build_personal_context_preview_snapshot`: that path omits Canvas provider/authority inputs, while dispatch supplies both. The shared first-request planner includes Canvas schemas and runtime guidance only when that provider is registered. This source mismatch exists in both dev `89dd84943abe8feacc5320d27a5c076641b83893` and original PR head `c945cf0278f5793417b7c9fd43201ed797a921fa`; it is not introduced by their merge. A native production-seam reproduction is still required by AC #1; the existing direct-planner Canvas test does not cover that seam.

The current Canvas owner requires `register_run` before its tools advertise. Registration creates run/staging ownership, so reusing it during disposable inspection is not an acceptable budget workaround. Read ADR-121 and ADR-186 before selecting the preview contract; record the ADR decision in the implementation plan before coding.

The task was created with Backlog CLI, then its offered ID 33126 was corrected before linking: an all-ref/worktree sweep found the current ceiling at 33164 (92 remote refs and 88 worktrees).
