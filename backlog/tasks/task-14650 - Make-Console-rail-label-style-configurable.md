---
id: TASK-14650
title: Make Console rail label style configurable
status: In Progress
assignee: []
created_date: '2026-08-08 05:44'
updated_date: '2026-08-08 15:23'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-08-07-console-rail-label-setting-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users keep the existing horizontal collapsed Console rail labels or opt into the compact stacked presentation from the canonical Settings screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Missing, false, or malformed preferences render the established horizontal labels: `Context ▸` at width 13 and `Inspector` at width 11
- [x] #2 Settings > Console Behavior provides a searchable, keyboard-operable opt-in for stacked three-column collapsed rail labels
- [x] #3 The setting exposes its saved or unsaved Horizontal/Stacked state in text and provides focused guidance describing its consequence and timing
- [x] #4 Save persists all Console Behavior drafts and updates the in-memory preference only after persistence succeeds; failed Save retains the draft and active style
- [x] #5 Revert discards all Console Behavior drafts together and reports the restored rail-label style
- [x] #6 The preference persists and reloads across app sessions, and the shipped configuration default is horizontal
- [x] #7 Returning to a freshly constructed Console immediately reflects a successful Save without restarting Chatbook; failed Save retains the prior style
- [x] #8 Expanded headers, readable tooltips, badges, focus behavior, open/collapse behavior, and non-Console rails remain unchanged in both styles
- [x] #9 Settings and Console user guides document the default, both styles, Save/Revert scope, and when changes become visible
- [x] #10 Configuration, Settings, Console rail, mounted geometry/paint, and Settings-to-Console lifecycle tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the normalized config default and configuration tests.
2. Make fresh Console composition choose horizontal or stacked rails from app_config and verify exact labels, widths, tooltips, and rail behavior.
3. Add the staged searchable Settings control with text state, focused guidance, category-wide Save/Revert/failure feedback, and success-only runtime updates.
4. Prove Settings-to-fresh-Console lifecycle behavior, update both user guides, run visual and regression verification, and complete PR hygiene.

ADR required: no
ADR path: N/A
Reason: additive presentation preference using existing config, Settings, and Console boundaries.

Detailed plan: Docs/superpowers/plans/2026-08-08-console-rail-label-setting.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented an opt-in stacked presentation for collapsed Console rail labels while preserving Horizontal as the compatibility default.

- Added console.stack_collapsed_rail_labels normalization and the shipped false default. Missing, malformed, false, true, and string-boolean values are covered.
- Fresh Console composition resolves the preference once for both handles. Horizontal remains Context ▸ at width 13 and Inspector at width 11; Stacked uses width 3 and preserves tooltips, badges, focus, and open/collapse behavior.
- Added a searchable, keyboard-operated Rail presentation checkbox to canonical Settings > Console Behavior with text-carried saved/unsaved state, focused consequence/timing guidance, category-wide Save/Revert semantics, dynamic success/failure feedback, and success-only runtime activation.
- Added Settings-to-fresh-Console lifecycle coverage for successful and failed saves, plus exact persistence payload, field search, keyboard, Revert, ownership, geometry, tooltip, badge, and interaction coverage.
- Updated Docs/User_Guide/settings.md and Docs/User_Guide/console/chat-basics.md. Hardened the design after Impeccable critique and added the implementation plan.
- Visual verification used isolated Textual test harnesses at 190x55 for Settings and 160x45 for both Console styles. Captures are under tmp/task3401-visuals.
- Fresh focused verification: 42 feature/config/rail/lifecycle tests passed; 23 existing Console Behavior tests passed. Python compilation and git diff --check passed.
- Broader targeted run before the final ownership expectation correction recorded 384 passes and 9 failures; the feature-caused ownership failure was corrected and reverified. The remaining eight are existing dev baselines: four 120x30 Console geometry tests and four Appearance tests caused by the missing _appearance_bool_label helper.
- Repository-wide pytest was attempted and stopped during collection with 28 missing optional-dependency errors, including NumPy, Playwright, and audio/TTS extras. Ruff and MyPy are not installed in this worktree, so strict Definition of Done remains blocked and the task stays In Progress.
- ADR required: no. This is an additive presentation preference using existing config, Settings, and Console boundaries; no schema, dependency, security, or cross-module contract decision was introduced.
- Backlog hygiene: renumbered from TASK-3401 to TASK-14650 in PR #1465 because dev also contained the established TASK-3401 video-generation epic and the duplicate-ID guard correctly rejected the collision.

Modified production areas: tldw_chatbook/config.py, tldw_chatbook/UI/Screens/chat_screen.py, tldw_chatbook/UI/Screens/settings_screen.py. Added or updated focused tests, user guides, design/plan artifacts, and this Backlog task.
<!-- SECTION:NOTES:END -->
