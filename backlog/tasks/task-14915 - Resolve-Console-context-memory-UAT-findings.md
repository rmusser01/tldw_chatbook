---
id: TASK-14915
title: Resolve Console context-memory UAT findings
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 04:39'
updated_date: '2026-08-11 05:06'
labels:
  - console
  - ux
  - context
dependencies: []
references:
  - backlog/decisions/052-console-conversation-memory-and-compaction-policy.md
  - Docs/superpowers/qa/console-context-memory-uat-2026-08-10/README.md
modified_files:
  - tldw_chatbook/Chat/console_session_settings.py
  - tldw_chatbook/Widgets/Console/console_context_controls.py
  - tldw_chatbook/Widgets/Console/console_model_popover.py
  - tldw_chatbook/Widgets/Console/console_settings_modal.py
  - tldw_chatbook/UI/Screens/settings_screen.py
  - Tests/Chat/test_console_session_settings.py
  - Tests/UI/test_console_context_controls.py
  - Tests/UI/test_console_session_settings.py
  - Tests/UI/test_settings_configuration_hub.py
  - Tests/UI/test_settings_context_memory_controls.py
  - Docs/superpowers/qa/console-context-memory-uat-2026-08-10/README.md
  - >-
    Docs/superpowers/qa/console-context-memory-uat-2026-08-10/uat_context_memory.py
  - Docs/superpowers/qa/console-context-memory-uat-2026-08-10/captures
priority: high
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Console context and memory journey trustworthy, discoverable, and understandable for new users across wide and narrow terminals, while preserving safe request admission.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unknown-model fallback is 8001 tokens and visibly labeled estimated or unverified
- [x] #2 Conversation save scope and model-default scope are unambiguous
- [x] #3 Response max tokens and Conversation max tokens are visibly distinct concepts
- [x] #4 Quick compaction controls explain summarization and potential extra model calls
- [x] #5 Quick settings exposes hidden content and keeps actions reachable on narrow terminals
- [x] #6 Context modal exposes hidden content and focuses the requested section on narrow terminals
- [x] #7 Global context controls are easier to discover in Console Behavior
- [x] #8 Advanced compaction labels describe outcomes without relying on the inspector
- [x] #9 Genuine-overflow recovery copy remains user-goal-first
- [x] #10 Automated and step-by-step UAT evidence covers all findings
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace capacity provenance, labels, and save-scope copy across quick and full Console settings. 2. Implement the 8001-token fallback and visibly unverified provenance. 3. Distinguish response-output and conversation-length limits. 4. Add concise compaction consequences and narrow-terminal fold/action affordances. 5. Improve canonical Settings ordering and outcome-oriented labels. 6. update the UAT register and automated coverage, then rerun wide/narrow UAT. ADR required: no. ADR path: backlog/decisions/052-console-conversation-memory-and-compaction-policy.md. Reason: this aligns the existing accepted UI and provenance contract without changing ownership or behavior boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented all nine UAT resolutions: an 8001-token visibly unverified fallback, explicit conversation versus model-default scope, separate response-output and conversation-length controls, compaction consequence copy, narrow-terminal scrolling/focus affordances, improved global discovery, and outcome-oriented labels. Fixed the full-modal scroll root cause by allowing the nested context view to expose its intrinsic height. ADR required: no; ADR-052 already defines the ownership and safety contract. Verification: 100 targeted tests passed, Ruff passed for changed files (with only pre-existing Settings import/type-comparison ignores), git diff --check passed, and the refreshed wide/narrow full-app UAT passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved all nine Console context-memory UAT findings and changed the unknown-model fallback to 8001 tokens without conflating it with the independent response max. Updated automated tests and step-by-step UAT evidence.
<!-- SECTION:FINAL_SUMMARY:END -->
