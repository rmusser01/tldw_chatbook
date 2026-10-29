---
id: TASK-37600
title: Keep character chat model and workspace activation consistent
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-29 20:04'
updated_date: '2026-10-29 20:26'
labels:
  - console
  - roleplay
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Starting a character conversation can revive an older provider selection, and closing its model picker can return the user to a different workspace chat. Preserve the current chat model selection and the newly opened character conversation through modal resume.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Character launches use the current non-pristine chat provider and model even when older controls disagree, while untouched chats still use current defaults.
- [x] #2 Character chats opened from a named workspace remain active after model Apply or Cancel, with the registry aligned to the character chat.
- [x] #3 Existing control-derived fallback, prompt seeding, pristine-session reuse, and source conversation behavior remain covered.
- [x] #4 Targeted regressions and UAT exercise launch, Apply, Cancel and a real local HTTP reply, with evidence recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/095-conversation-owned-console-generation-settings.md; existing TASK-15120 workspace ownership ruling. Reason: restores existing conversation settings and workspace activation ownership without a new storage, security or service boundary. 1. Add failing character launch and modal resume regressions. 2. Prefer the active worked chat pair over stale control mirrors through one shared character-default helper. 3. Align the registry through the existing workspace seam at character activation. 4. Run targeted tests, static checks, actual-app UAT and independent review; record evidence and open a PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Character launch now resolves the current worked chat provider/model through one shared helper and the existing model-default chain; untouched/no-source fallback remains intact. Personas handoffs and Console picker New Chat align the actual target workspace through the existing seam, preserving the selected chat after modal resume. Source prompts and prefills remain separate. Changed session.py, character.py, wiring.py, the existing session-controller test module, architecture notes, and reproducible QA evidence. Six regressions failed before the fix; 24 character-launch checks and the final 19-case targeted set passed. Real served-app UAT passed launch, model Apply, backdrop Cancel and an HTTP reply using the selected model and character system prompt, with source messages unchanged. Independent code/evidence review found no Critical or Important issue. Five adjacent prompt-seed tests retain pre-existing config-fixture admission errors; one representative reproduced in the pre-fix checkout. Ruff introduced zero diagnostics over baseline; task-ID guard and diff hygiene passed. ADR required: no; existing ADR-006/095 and TASK-15120 apply. Evidence and limitations: Docs/superpowers/qa/2026-10-29-character-chat-selection/qualification.md.
<!-- SECTION:NOTES:END -->
