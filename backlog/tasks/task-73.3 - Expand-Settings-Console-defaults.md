---
id: TASK-73.3
title: Expand Settings Console defaults
status: Done
labels:
- settings
- console
- configuration
- ux
dependencies:
- TASK-73.1
- TASK-73.2
priority: high
parent_task_id: TASK-73
documentation:
- Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expand Console Behavior into a real global-defaults category for supported Console settings while preserving Console as the owner of active session and run state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console global defaults load from config and distinguish global fallbacks from provider+model defaults and per-session Console overrides.
- [x] #2 The `streaming` and `enable_streaming` compatibility seam has one documented effective source of truth before new controls are added.
- [x] #3 Supported global fallback defaults can be edited, validated, saved, reverted, and reflected in Console behavior when no provider+model default or session override applies.
- [x] #4 Invalid numeric or boolean values are blocked with visible recovery copy.
- [x] #5 Existing large-paste collapse behavior remains intact and covered.
- [x] #6 Mounted tests verify save/revert and Console reflection for changed defaults.
- [x] #7 Actual CDP/Textual-web screenshot QA verifies the Console Behavior category and is approved before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing pure and mounted regressions for effective Console default loading, validation, save/revert, and Console reflection.
2. Define canonical `chat_defaults.streaming` behavior while preserving `enable_streaming` as a compatibility fallback.
3. Add guided Console Behavior controls for supported global fallback defaults without changing provider+model profile ownership or active session override ownership.
4. Persist validated defaults through `SettingsConfigAdapter`, refresh runtime defaults, and keep large-paste behavior intact.
5. Run focused verification, capture actual Textual-web/CDP screenshots, record approval, and close task evidence before PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added canonical Console fallback loading for `chat_defaults.streaming`, with legacy `enable_streaming` treated only as a compatibility fallback when `streaming` is absent.
- Expanded Settings > Console Behavior with editable global fallbacks for streaming, temperature, top_p, max_tokens, and the existing Console composer paste-collapse settings.
- Added validation and visible recovery copy for invalid boolean and numeric values, plus save/revert coverage through a batched `SettingsConfigAdapter.save_sections(...)` persistence path.
- Added a single-write config helper for multi-section saves so Console Behavior no longer performs per-key TOML read/write/reload calls on the UI thread.
- Preserved active Console session and provider+model profile ownership: the new settings are global fallbacks, not replacements for per-session or provider/model overrides.
- Reworked the Console Behavior inspector into a category-specific control guide that explains each visible setting and its precedence instead of generic Settings ownership copy.
- Added pure and mounted regressions covering effective default loading, legacy compatibility, invalid value rejection, save/revert behavior, large-paste collapse controls, and inspector guidance.
- Captured actual Textual-web/CDP screenshots for the full-width Settings screen and the Console Behavior inspector guide:
  - `Docs/superpowers/qa/settings-configuration-hub/stage-3-console-defaults-full-width.png`
  - `Docs/superpowers/qa/settings-configuration-hub/stage-3-console-defaults-inspector-control-guide.png`
- User approved the rendered Settings screenshot after the inspector guidance update.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Settings Console Behavior now acts as a real configuration hub category for Console global fallbacks, while preserving Console session state and provider/model profile precedence. The visible Settings inspector now explains the selected category's controls directly, and focused tests plus CDP screenshot QA cover the implemented behavior.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Unit, mounted, and integration tests relevant to changed behavior pass
- [x] #3 Static analysis and diff hygiene checks pass
- [x] #4 Actual app QA walkthrough completed with screenshots
- [x] #5 User approval recorded for visible Settings changes
- [x] #6 Documentation and task notes updated
- [x] #7 Task status moved to Done after implementation notes are added
<!-- DOD:END -->
